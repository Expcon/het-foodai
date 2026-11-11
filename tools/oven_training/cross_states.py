"""Read-only W/S counterfactuals gated on exact R0 reproduction, not model selection."""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path

from .model import configure_sdk, digest, frame_image, new_model, output_dir
from .monitor import FixedImages, bn_snapshot, evaluate_fixed, model_state_sha256
from .observation import atomic_new, load_snapshot, mix_states, write_json_new
from .round2 import resolve_image


def transition_summary(before, after, keys):
    if not before or len(before)!=len(after) or not keys: raise ValueError('Paired nonempty predictions required')
    rows=[]
    for a,b in zip(before,after):
        if any(a[k]!=b[k] for k in ['sample_index','target_id','input_sha256']):
            raise ValueError('Counterfactual image/target/input pairing differs')
        if any(type(i) is not int or not 0<=i<len(keys) for i in [a['target_id'],a['top1_id'],b['top1_id']]):
            raise ValueError('Invalid counterfactual class index')
        rows.append((a['target_id'],a['top1_id'],b['top1_id']))
    def counts(values):
        return dict(samples=len(values),before_correct=sum(t==a for t,a,b in values),
                    after_correct=sum(t==b for t,a,b in values),rescued=sum(a!=t and b==t for t,a,b in values),
                    introduced=sum(a==t and b!=t for t,a,b in values),changed_predictions=sum(a!=b for t,a,b in values),
                    changed_wrong_to_wrong=sum(a!=t and b!=t and a!=b for t,a,b in values))
    return dict(**counts(rows),per_class={key:counts([r for r in rows if r[0]==i]) for i,key in enumerate(keys)})


def delta_evidence(model, first, second):
    import torch
    bn_affine={name+('.' if name else '')+field for name,module in model.named_modules()
               if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for field in ['weight','bias']}
    groups={key:dict(tensors=0,changed_tensors=0,delta_squared=0.) for key in ['linear_head','head_conv','bn_affine','other_backbone']}
    for name,_ in model.named_parameters():
        group='bn_affine' if name in bn_affine else 'linear_head' if name.startswith('model.10.linear.') else 'head_conv' if name.startswith('model.10.') else 'other_backbone'
        delta=second[name].double()-first[name].double();g=groups[group]
        g['tensors']+=1;g['changed_tensors']+=int(not torch.equal(first[name],second[name]));g['delta_squared']+=float(delta.square().sum())
    for g in groups.values():g['delta_l2']=g.pop('delta_squared')**.5
    return groups


def run_cross(observations, rows_path, source_root, out, device='cuda:0'):
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image
    observations=Path(observations);out=output_dir(out)
    gate=json.loads((observations/'gate.json').read_text(encoding='utf-8'))
    if (gate.get('status'),gate.get('reference_bound'),gate.get('all_reference_metrics_and_batches_equal'),gate.get('epochs'),gate.get('global_steps')) != ('PASS',True,True,30,1890):
        raise ValueError('Exact real R0 gate must pass before counterfactual attribution')
    binding=json.loads((observations/'binding.json').read_text(encoding='utf-8'))
    if digest(rows_path)!=binding['config']['rows_sha256']:raise ValueError('Cross experiment data manifest changed')
    rows=[r for r in json.loads(Path(rows_path).read_text(encoding='utf-8')) if r['new_split']=='train']
    keys=binding['config']['class_keys']
    if len(rows)!=len(binding['sources']):raise ValueError('Cross data sample count differs')
    items=[]
    for i,(row,source) in enumerate(zip(rows,binding['sources'])):
        path=resolve_image(row,source_root)
        if source['sample_index']!=i or source['class_key']!=row['class_key'] or source['source_sha256']!=row['sha256'] or digest(path)!=row['sha256']:
            raise ValueError('Cross source/order/label binding differs')
        with Image.open(path) as image:items.append((frame_image(image),keys.index(row['class_key'])))
    index=[json.loads(s) for s in (observations/'checkpoints.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(index)!=41 or len({r['file'] for r in index})!=41:raise ValueError('Incomplete/duplicate observation index')
    archive={r['file']:r['sha256'] for r in index}
    for name,sha in archive.items():
        path=(observations/name).resolve()
        if not path.is_relative_to(observations.resolve()) or digest(path)!=sha:raise ValueError('Unsafe/changed checkpoint index')
    out.mkdir(parents=True,exist_ok=False);configure_sdk(out/'sdk-settings')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    template=new_model(keys).cpu()
    loader=DataLoader(FixedImages(items),batch_size=32,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(10042))
    cache={}
    def checkpoint(name):
        filename='checkpoints/'+name
        if filename not in cache:
            payload=load_snapshot(observations/filename,archive[filename])
            if payload['binding']!=binding:raise ValueError('Checkpoint binding differs')
            attrs={name:{k:v[k] for k in ['eps','momentum','track_running_stats']} for name,v in bn_snapshot(template)['layers'].items()}
            if attrs!=payload['bn_attributes']:raise ValueError('Counterfactual BN attributes differ')
            cache[filename]=payload
        return cache[filename]
    pairs=[('epoch_27_28','epoch_027.pt','epoch_028.pt'),('epoch_29_30','epoch_029.pt','epoch_030.pt')]
    pairs += [(f'tail_{e:02d}',f'epoch_{e:03d}_tail_pre.pt',f'epoch_{e:03d}_tail_post.pt') for e in [24,27,28,29,30]]
    summaries=[];class_table=[];all_metrics=[]
    for pair_name,left_name,right_name in pairs:
        left=checkpoint(left_name);right=checkpoint(right_name)
        if pair_name.startswith('tail_'):
            end=checkpoint(f'epoch_{right["epoch"]:03d}.pt')
            if right['model_state_sha256']!=end['model_state_sha256']:raise ValueError('Post-tail state differs from epoch end')
            if len(left['pending'][1])!=12 or right['global_step']-left['global_step']!=1:raise ValueError('Wrong tail boundary')
        paired=[left,right];evaluations={};evidence={}
        for w,s in [(0,0),(0,1),(1,0),(1,1)]:
            label=f'W{w}_S{s}'
            model=copy.deepcopy(template).to(device).eval()
            model.load_state_dict(mix_states(model,paired[w]['model_state'],paired[s]['model_state']),strict=True)
            fingerprint=model_state_sha256(model);per_image=[]
            metrics=evaluate_fixed(model,loader,keys,device,per_image=per_image)
            if model_state_sha256(model)!=fingerprint:raise ValueError('Counterfactual eval mutated persistent state')
            # Same-epoch diagonal states must reproduce the recorded historical fixed eval.
            if w==s and paired[w]['phase']!='pre_forward':
                historical=json.loads((observations/'eval'/f'epoch_{paired[w]["epoch"]:03d}_metrics.json').read_text(encoding='utf-8'))
                if metrics!=historical:raise ValueError('Diagonal eval does not reproduce R0')
            for row,source in zip(per_image,binding['sources']):row.update(source)
            evaluations[label]=metrics;evidence[label]=per_image
            target=out/pair_name/label
            write_json_new(target/'metrics.json',dict(metrics,weights_checkpoint=paired[w]['model_state_sha256'],
                statistics_checkpoint=paired[s]['model_state_sha256'],mixed_state_sha256=fingerprint,scope='mechanism-only seen-training-image counterfactual; not deployable'))
            data=''.join(json.dumps(r,allow_nan=False)+'\n' for r in per_image).encode('utf-8')
            atomic_new(target/'per_image.jsonl',lambda stream:stream.write(data))
            all_metrics.append(dict(pair=pair_name,combination=label,correct=metrics['correct'],samples=metrics['samples'],loss=metrics['loss']))
            for key,v in metrics['per_class'].items():class_table.append(dict(pair=pair_name,combination=label,class_key=key,**v))
            print(json.dumps(dict(pair=pair_name,combination=label,correct=metrics['correct'],loss=metrics['loss'])),flush=True)
            del model
        transitions={}
        for name,a,b in [('historical','W0_S0','W1_S1'),('S_at_W0','W0_S0','W0_S1'),('S_at_W1','W1_S0','W1_S1'),('W_at_S0','W0_S0','W1_S0'),('W_at_S1','W0_S1','W1_S1')]:
            transitions[name]=transition_summary(evidence[a],evidence[b],keys)
            changes=[dict(sample_index=l['sample_index'],source_sha256=l['source_sha256'],input_sha256=l['input_sha256'],target_id=l['target_id'],
                          before_top1_id=l['top1_id'],after_top1_id=r['top1_id'],before_loss=l['loss'],after_loss=r['loss']) for l,r in zip(evidence[a],evidence[b])]
            data=''.join(json.dumps(r)+'\n' for r in changes).encode('utf-8')
            atomic_new(out/pair_name/(name+'_transitions.jsonl'),lambda stream:stream.write(data))
        result=dict(pair=pair_name,left=left_name,right=right_name,metrics=evaluations,transitions=transitions,
            parameter_delta=delta_evidence(template,left['model_state'],right['model_state']),
            scope='within-trajectory counterfactual; effects can interact; no unique causal percentage or generalization claim')
        write_json_new(out/pair_name/'comparison.json',result);summaries.append(result)
    for name,values in [('combinations.csv',all_metrics),('per_class.csv',class_table)]:
        import io
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        atomic_new(out/name,lambda target:target.write(stream.getvalue().encode('utf-8-sig')))
    for name,sha in archive.items():
        if digest(observations/name)!=sha:raise ValueError('Counterfactual altered an observation file')
    final=dict(status='PASS',scope='seen-training-image mechanism diagnostics only; not independent accuracy or deployment',
               class_order=keys,samples=len(rows),pairs=summaries,checkpoint_files_unchanged=len(archive),device=device)
    write_json_new(out/'summary.json',final)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    run_cross(args.observations,args.rows,args.source_root,args.out,args.device)


if __name__=='__main__':main()
