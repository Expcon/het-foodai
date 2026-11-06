"""Read-only external oven data: v1 diagnosis before fixed exploratory joint fit."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path, PurePosixPath
import time

from .model import ROOT,PREPROCESS,configure_sdk,digest,load_model,output_dir,transform
from .export_desk import write_json

V1_KEYS=['asparagus','bacon','broccoli','chicken_wing','croissant','french_fries','potato','salmon','toast']
V2_KEYS=V1_KEYS+['corn','sausage','steak']
V1_SHA='72367305b65b31920c59865758dcdd4a1ee506751bde57db113f017c5c314b25'


def resolve_image(row, source_root):
    relative=row.get('relative_path')
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative or '..' in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
        raise ValueError('Unsafe relative image path')
    if row.get('source')=='new':root=Path(source_root).resolve()
    elif row.get('source')=='round1':
        root=ROOT.resolve()
        if not relative.startswith('data/raw/oven_round1_20261106/'):raise ValueError('Only original round1 raw data allowed')
    else:raise ValueError('Unknown source root')
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root:raise ValueError('Image escapes its approved root')
    return path


def diagnostic_metrics(records, keys):
    supported=[r for r in records if r['class_key'] in keys]
    trusted=[r for r in supported if r.get('trusted_label') is True]
    def stats(rows, accuracy):
        matrix=[[0]*len(keys) for _ in keys]
        for r in rows:matrix[keys.index(r['class_key'])][keys.index(r['top3'][0]['class_key'])]+=1
        n=len(rows);correct=sum(matrix[i][i] for i in range(len(keys)))
        result={'count':n,'correct':correct,'class_order':keys,'confusion':matrix,
                'per_class':{k:{'count':sum(matrix[i]),'correct':matrix[i][i],
                               ('accuracy' if accuracy else 'directory_agreement_fraction'):matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None,
                               'small_sample':sum(matrix[i])<30} for i,k in enumerate(keys)}}
        if accuracy:result['accuracy']=correct/n if n else None
        else:result['directory_agreement_fraction']=correct/n if n else None
        return result
    return {'trusted_supported':stats(trusted,True),'candidate_directory_agreement_not_accuracy':stats(supported,False),
            'unsupported_images':len(records)-len(supported),'independent_test_accuracy':None}


def summarize_diagnostic(records,keys):
    summary=diagnostic_metrics(records,keys)
    summary['scope']='diagnostic only; unreviewed directory agreement is NOT accuracy; not final independent test'
    summary['strata']={}
    for field in ['foil','shelf','oven_id','session_id','lighting','angle','source_collection']:
        groups=defaultdict(list)
        for r in records:
            value=r.get(field)
            if isinstance(value,dict):value=value.get('value')
            groups[str(value or 'unknown')].append(r)
        summary['strata'][field]={k:diagnostic_metrics(v,keys) for k,v in sorted(groups.items())}
    summary['forced_unknown_predictions']={}
    for key in sorted({r['class_key'] for r in records if r['class_key'] not in keys}):
        rows=[r for r in records if r['class_key']==key]
        summary['forced_unknown_predictions'][key]={'count':len(rows),'predicted_counts':dict(Counter(r['top3'][0]['class_key'] for r in rows)),
            'score_at_least_0_9':sum(r['top3'][0]['score']>=.9 for r in rows),'is_ood_detector':False}
    return summary


def image_proxies(image):
    """Photometric descriptors; never a food mask or target-size estimate."""
    import numpy as np
    rgb=np.asarray(image.convert('RGB').resize((256,144)),dtype=np.float32)/255
    gray=rgb.mean(axis=2)
    return {'mean_intensity':float(gray.mean()),'intensity_std':float(gray.std()),
            'bright_neutral_fraction':float(((gray>.9)&(rgb.max(2)-rgb.min(2)<.1)).mean()),
            'dark_fraction':float((gray<.1).mean()),'food_pixel_fraction':None,
            'note':'Whole-frame brightness proxies are not reflectance/material/food segmentation evidence'}


def diagnose(args):
    import torch
    from PIL import Image,ImageOps
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    configure_sdk(out/'sdk-settings')
    torch.set_num_threads(4)
    rows=json.loads(args.rows.read_text(encoding='utf-8'))
    new=[r for r in rows if r['source']=='new']
    if not new:raise ValueError('No new photos to diagnose')
    model,meta=load_model(args.checkpoint,args.device,args.expect_sha256)
    keys=meta['class_keys'];pre=transform(False);records=[];failures=[];timings=[]
    for offset in range(0,len(new),16):
        tensors=[];batch=[]
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        for row in new[offset:offset+16]:
            try:
                path=resolve_image(row,args.source_root)
                if digest(path)!=row['sha256']:raise ValueError('Audited image bytes changed')
                with Image.open(path) as image:
                    rgb=ImageOps.exif_transpose(image).convert('RGB')
                    tensors.append(pre(rgb));features=image_proxies(rgb)
                batch.append((row,features))
            except (OSError,ValueError) as exc:
                failures.append({'image_id':row['image_id'],'error':str(exc)})
        if tensors:
            with torch.inference_mode():probs=model(torch.stack(tensors).to(args.device))[0].cpu()
            if not torch.isfinite(probs).all():raise ValueError('Nonfinite model output')
            values,indices=probs.topk(3,dim=1)
            for (row,features),p,vs,ids in zip(batch,probs.tolist(),values.tolist(),indices.tolist()):
                record={k:row.get(k) for k in ['image_id','relative_path','sha256','class_key','class_id','trusted_label','label_evidence','group_id','physical_sample_id','foil','shelf','oven_id','session_id','lighting','angle','source_collection']}
                record.update(top3=[{'class_id':i,'class_key':keys[i],'score':v} for i,v in zip(ids,vs)],probabilities=p,features=features)
                records.append(record)
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        timings.append({'batch_count':len(batch),'decode_preprocess_model_proxies_ms':1000*(time.perf_counter()-start)})
        if offset%160==0:print(f'Diagnosed {min(offset+16,len(new))}/{len(new)}',flush=True)
    result={'state':'DIAGNOSTIC_COMPLETED','checkpoint_sha256':digest(args.checkpoint),'rows_sha256':digest(args.rows),
            'seen_during_training':bool(meta.get('config',{}).get('provenance',{}).get('dataset_version')=='oven-round2-20261107'),
            'class_keys':keys,'images':len(records),'failures':failures,'metrics':summarize_diagnostic(records,keys),
            'device':args.device,'timing_scope':'hash/decode/EXIF/preprocess/photometric proxies/H2D/forward/D2H; CUDA synchronized; no model load',
            'batch_timings':timings}
    write_json(out/'summary.json',result)
    write_json(out/'predictions.json',records)
    write_json(out/'trusted_label_errors.json',[r for r in records if r['trusted_label'] is True and r['class_key'] in keys and r['top3'][0]['class_key']!=r['class_key']])
    write_json(out/'directory_disagreements_not_verified_errors.json',[r for r in records if r['class_key'] in keys and r['top3'][0]['class_key']!=r['class_key']])
    print(json.dumps({'images':len(records),'processing_failures':len(failures),**diagnostic_metrics(records,keys)},ensure_ascii=False),flush=True)
    return 0 if not failures else 2


def require_complete_v1_diagnostic(diagnostic, rows, rows_sha):
    count=sum(r.get('source')=='new' for r in rows)
    if (diagnostic.get('state')!='DIAGNOSTIC_COMPLETED' or diagnostic.get('checkpoint_sha256')!=V1_SHA or
        diagnostic.get('rows_sha256')!=rows_sha or diagnostic.get('failures')!=[] or
        diagnostic.get('class_keys')!=V1_KEYS or count==0 or diagnostic.get('images')!=count):
        raise ValueError('Complete v1 diagnosis of this audited version is required before training')


def train(args):
    from .train import fit_exploratory
    diagnostic=json.loads(args.diagnostic.read_text(encoding='utf-8'))
    rows=json.loads(args.rows.read_text(encoding='utf-8'))
    require_complete_v1_diagnostic(diagnostic,rows,digest(args.rows))
    active=[];seen=set()
    for r in rows:
        if r['new_split']=='excluded':continue
        if r['new_split']!='train' or not r.get('group_id') or r['class_key'] not in V2_KEYS or r['class_id']!=V2_KEYS.index(r['class_key']):
            raise ValueError('Exploratory fit accepts grouped train rows with explicit append-only IDs')
        path=resolve_image(r,args.source_root)
        if digest(path)!=r['sha256']:raise ValueError('Original image changed since audit')
        if r['pixel_sha256'] in seen:raise ValueError('Unremoved pixel duplicate in training')
        seen.add(r['pixel_sha256']);active.append((path,r['class_key']))
    if set(key for _,key in active)!=set(V2_KEYS):raise ValueError('All12 experimental classes require data')
    provenance={'dataset_version':'oven-round2-20261107','v1_diagnostic_sha256':digest(args.diagnostic),
                'label_policy':'organizer-directory weak labels; pending JSON labels NOT treated as human truth',
                'holdout':'none; no proven independent specimens; final_test_not_defined',
                'per_class_samples':dict(Counter(k for _,k in active)),
                'v1_checkpoint_preserved_sha256':V1_SHA,'scope':'fixed exploratory joint fit, not a formal controlled accuracy comparison'}
    fit_exploratory(active,V2_KEYS,args.out,args.pretrained,args.device,digest(args.rows),provenance)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['diagnose','train'])
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expect-sha256')
    p.add_argument('--diagnostic',type=Path)
    p.add_argument('--pretrained',type=Path)
    args=p.parse_args()
    if args.action=='diagnose' and (not args.checkpoint or not args.expect_sha256):p.error('Diagnosis requires checkpoint and SHA')
    if args.action=='train' and (not args.diagnostic or not args.pretrained):p.error('Training requires completed v1 diagnostic and fixed upstream initialization')
    return diagnose(args) if args.action=='diagnose' else train(args)


if __name__=='__main__':raise SystemExit(main())
