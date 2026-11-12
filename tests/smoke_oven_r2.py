"""Actual YOLO CUDA SYNTHETIC34/35 tail batches and safe exact pending replay."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset
from tools.oven_training.model import configure_sdk,digest,new_model,output_dir,PRETRAINED_SHA
from tools.oven_training.round2 import V2_KEYS
from tools.oven_training.r2 import R2_RECIPE
from tools.oven_training.train import fit_exploratory
from tools.oven_training.observation import load_snapshot,restore_training,state_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True)
    parser.add_argument('--reference-initialization',type=Path,required=True);args=parser.parse_args()
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Pinned official pretrained required')
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    expected=json.loads(args.reference_initialization.read_text('utf-8'))['model_state_sha256']
    generator=np.random.default_rng(20261112);all_entries=[];results=[]
    for i in range(67):
        path=out/f'SYNTHETIC_{i:03d}.png';Image.fromarray(generator.integers(0,256,(90,160,3),dtype=np.uint8)).save(path)
        all_entries.append((path,V2_KEYS[i%12]))
    for n,tail in [(66,34),(67,35)]:
        run=out/f'SYNTHETIC_{n}'
        fit_exploratory(all_entries[:n],V2_KEYS,run,args.pretrained,'cuda:0','a'*64,
            provenance={'scope':'SYNTHETIC SOFTWARE ONLY; not food accuracy','reference_initialization_state_sha256':expected},
            recipe={**R2_RECIPE,'epochs':1},r2_control=True,observation={'comparison':'record_only','tail_epochs':(1,)})
        monitor=json.loads((run/'monitor/epoch_001.json').read_text('utf-8'))
        assert monitor['batch_trace']['sizes']==[32,tail]
        assert sorted(i for b in monitor['batch_trace']['batches'] for i in b['indices'])==list(range(n))
        assert all(r['counter_delta']==2 for r in monitor['bn_update']['layers'].values())
        assert all(r['finite'] for r in monitor['bn']['layers'].values())
        obs=run/'observations';index={Path(r['file']).name:r for r in [json.loads(s) for s in (obs/'checkpoints.jsonl').read_text().splitlines()]}
        def read(name):
            record=index[name];return load_snapshot(obs/record['file'],record['sha256'])
        pre=read('epoch_001_tail_pre.pt');post=read('epoch_001_tail_post.pt')
        configure_sdk(out/'replay-sdk');model=new_model(V2_KEYS).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0005)
        empty=TensorDataset(torch.empty(0,3,224,224),torch.empty(0,dtype=torch.long))
        loaders={name:DataLoader(empty,generator=torch.Generator()) for name in pre['loader_generators']}
        restore_training(pre,model,opt,loaders,pre['binding'])
        images,labels,indices=pre['pending'];assert len(labels)==tail
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(images.cuda()),labels.cuda());loss.backward();opt.step()
        assert state_digest(model.state_dict())==post['model_state_sha256']
        for i,values in post['optimizer_state']['state'].items():
            for key,value in values.items():assert torch.equal(opt.state_dict()['state'][i][key].cpu(),value)
        results.append(dict(samples=n,batch_sizes=[32,tail],all_samples_once=True,BN_finite=True,BN_counter_delta=2,
            safe_full_states=len(index),tail_replay_model_optimizer_exact=True,initialization_equals_R1=True))
        del model,opt;torch.cuda.empty_cache()
    report=dict(status='PASS',scope='SYNTHETIC SOFTWARE ONLY',gpu=torch.cuda.get_device_name(),cases=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
