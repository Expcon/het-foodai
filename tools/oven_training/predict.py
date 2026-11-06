"""Offline experimental Top-3, batch evidence and explicit errors; no hardware access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from .model import ROOT,configure_sdk,digest,input_path,load_model,output_dir,transform


def summarize(records,keys):
    labelled=[r for r in records if r.get('expected') in keys]
    correct=sum(r['expected']==r['top3'][0]['class_key'] for r in labelled)
    confusion=[[0]*len(keys) for _ in keys]
    per_class={}
    for row in labelled:confusion[keys.index(row['expected'])][keys.index(row['top3'][0]['class_key'])]+=1
    for i,key in enumerate(keys):
        count=sum(confusion[i]);per_class[key]={'count':count,'correct':confusion[i][i],'recall_on_seen_images':confusion[i][i]/count if count else None}
    return dict(scope='resubstitution_diagnostic_not_independent_evaluation',labelled_images=len(labelled),correct=correct,
        resubstitution_accuracy=correct/len(labelled) if labelled else None,independent_validation_accuracy=None,
        class_order=keys,confusion_matrix=confusion,per_class=per_class)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expect-sha256',required=True,help='Full SHA from local run_info.json or trusted handoff')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--rows',type=Path)
    source.add_argument('--images',type=Path,nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args(argv)
    try:
        if not 1<=args.batch_size<=256:raise ValueError('batch-size must be 1..256')
        if not args.checkpoint.is_file():raise ValueError('model not ready: checkpoint missing')
        if len(args.expect_sha256)!=64 or digest(args.checkpoint)!=args.expect_sha256:
            raise ValueError('Checkpoint full SHA256 does not match trusted handoff')
        out=output_dir(args.out)
        if out.exists():raise ValueError('Use a new output directory')
        if args.rows:
            rows=json.loads(args.rows.read_text(encoding='utf-8'))
            entries=[(input_path(r['relative_path']),r.get('class_key'),r.get('original_split'),r.get('sha256'),r.get('new_split')) for r in rows]
        else:entries=[(p,None,None,None,None) for p in args.images]
        if not entries:raise ValueError('No input images')
        configure_sdk()
        import torch
        from PIL import Image
        sys.path.insert(0,str(ROOT/'src'))
        from foodai.predictor import validate_image_file
        torch.set_num_threads(4)
        for path,_,_,expected,_ in entries:
            validate_image_file(path)
            if expected and digest(path)!=expected:raise ValueError('Source image changed after audit')
        start=time.perf_counter()
        model,meta=load_model(args.checkpoint,args.device,args.expect_sha256)
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        load_ms=1000*(time.perf_counter()-start)
        process=transform(False);records=[];batches=[];keys=meta['class_keys']
        for offset in range(0,len(entries),args.batch_size):
            batch=entries[offset:offset+args.batch_size]
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            started=time.perf_counter();tensors=[]
            for path,_,_,_,_ in batch:
                with Image.open(path) as image:tensors.append(process(image))
            inputs=torch.stack(tensors).to(args.device)
            with torch.inference_mode():probs=model(inputs)[0]
            if not torch.isfinite(probs).all():raise ValueError('Non-finite predictions')
            scores,indices=probs.topk(min(3,len(keys)),dim=1)
            scores,indices=scores.cpu().tolist(),indices.cpu().tolist()
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            elapsed=1000*(time.perf_counter()-started)
            batches.append(dict(size=len(batch),batch_e2e_ms=elapsed,amortized_ms_per_image=elapsed/len(batch)))
            for (path,expected,original_split,_,new_split),vs,ids in zip(batch,scores,indices):
                records.append(dict(image=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
                    expected=expected,original_split=original_split,new_split=new_split,top3=[dict(class_id=i,class_key=keys[i],score=v) for i,v in zip(ids,vs)],
                    confidence_is_calibrated=False))
        result=dict(state='EXPLORATORY_PREDICTIONS_READY',device=args.device,checkpoint_sha256=digest(args.checkpoint),
            model_load_and_forward_check_ms=load_ms,image_count=len(records),metrics=summarize(records,keys),
            unique_training_images_metrics=summarize([r for r in records if r['new_split']=='train'],keys),
            timing=dict(scope='batch includes decode, EXIF/RGB, letterbox, tensor, H2D, forward, Top3, D2H; excludes initial validation/hash/model load/JSON writing',
                cuda_synchronized=args.device.startswith('cuda'),warmup='one zero-input forward in load verification; no image warmup',batches=batches),
            predictions=records)
        out.mkdir(parents=True)
        (out/'predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        failures=[r for r in records if r['expected'] in keys and r['expected']!=r['top3'][0]['class_key']]
        (out/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ['state','device','image_count','metrics']},ensure_ascii=False))
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
        print(json.dumps({'state':'ERROR_NOT_READY','reason':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
