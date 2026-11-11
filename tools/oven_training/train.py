"""Fixed-epoch, train-only exploratory YOLO fit. No automatic validation fallback."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import time

from .model import ROOT,PREPROCESS,class_digest,check_rows,configure_sdk,digest,input_path,new_model,output_dir,frame_image,transform


def fit_exploratory(entries, keys, out, pretrained, device, rows_sha256, provenance=None,
                    *, recipe=None, validation_entries=None, validation_report=None):
    """Shared fixed recipe; caller validates hashes, labels and train-only groups."""
    out=output_dir(out)
    if out.exists():raise ValueError('Use a new run directory; never overwrite a training run')
    if not entries or any(key not in keys for _,key in entries):raise ValueError('Explicit known training labels required')
    class_digest(keys)
    from .controlled import recipe_config
    from .monitor import (FixedImages, IndexedImages, evaluate_fixed, set_training_mode,
                          bn_snapshot, bn_delta, tensor_sha256, model_state_sha256)
    from .export_desk import write_json
    options=recipe_config(recipe)
    validation_entries=validation_entries or []
    validation_report=validation_report or {'status':'BLOCKED','issues':{'no_qualified_independent_validation':1}}
    if validation_entries and validation_report.get('status')!='READY_FOR_EXPLORATORY_MONITORING':
        raise ValueError('Validation entries require qualified independent identity evidence')
    if not validation_entries and validation_report.get('status')!='BLOCKED':
        raise ValueError('Missing validation cannot be marked ready')
    config=dict(**options,
        selection='final_epoch_only',validation=bool(validation_entries),independent_validation=False,
        rows_sha256=rows_sha256,pretrained_sha256=digest(pretrained),class_keys=keys,
        provenance=provenance or {})
    out.mkdir(parents=True)
    (out/'monitor').mkdir()
    (out/'fixed_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    configure_sdk(out/'sdk-settings')
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset,DataLoader
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
    class Images(Dataset):
        def __init__(self):
            self.items=[]
            for path,key in entries:
                with Image.open(path) as image:self.items.append((frame_image(image),keys.index(key)))
            self.augment=transform(True)
        def __len__(self):return len(self.items)
        def __getitem__(self,index):
            image,label=self.items[index]
            return self.augment(image),label
    dataset=Images()
    loader=DataLoader(IndexedImages(dataset),batch_size=32,shuffle=True,num_workers=0,
                      drop_last=False,generator=torch.Generator().manual_seed(42))
    fixed_loader=DataLoader(FixedImages(dataset.items),batch_size=32,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(10042))
    val_items=[]
    for path,key in validation_entries:
        if key not in keys:raise ValueError('Unknown validation class')
        with Image.open(path) as image:val_items.append((frame_image(image),keys.index(key)))
    val_loader=DataLoader(FixedImages(val_items),batch_size=32,shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(20042)) if val_items else None
    model=new_model(keys,pretrained).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    criterion=torch.nn.CrossEntropyLoss()
    initial=bn_snapshot(model)
    write_json(out/'monitor/initialization.json',dict(model_state_sha256=model_state_sha256(model),
        bn=initial,bn_policy=options['bn_policy'],pretrained_sha256=config['pretrained_sha256'],
        cache='deterministically framed PIL RGB images; augmentation sampled on each access',
        train_samples=len(entries),per_class=dict(Counter(key for _,key in entries)),
        timing_scope='epoch seconds include batch evidence and end-epoch eval; initialization and epoch0 eval excluded',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        training_source_sha256=digest(Path(__file__)),validation=validation_report))
    def finish_epoch(epoch, before, trace, tail=None):
        after=bn_snapshot(model);change=bn_delta(before,after)
        expected=len(loader) if epoch and options['bn_policy']=='update' else 0
        if any(layer['counter_delta']!=expected for layer in change['layers'].values()):
            raise RuntimeError('BN update count differs from configured policy')
        if options['bn_policy']=='frozen_stats' and not change['running_unchanged']:
            raise RuntimeError('Frozen BN running statistics changed')
        fixed=evaluate_fixed(model,fixed_loader,keys,device)
        validation={'status':validation_report['status'],'readiness':validation_report}
        if val_loader is not None:
            validation['metrics']=evaluate_fixed(model,val_loader,keys,device)
            validation['metrics']['scope']='declared independent, human-labelled experimental validation; not frozen final test'
        record=dict(epoch=epoch,train_eval=fixed,validation=validation,bn=after,bn_update=change,
                    batch_trace={'sizes':[r['size'] for r in trace],'batches':trace},last_batch=tail,
                    fixed_eval_scope='training resubstitution diagnostic; no model selection or formal accuracy')
        write_json(out/'monitor'/f'epoch_{epoch:03d}.json',record)
        return record
    finish_epoch(0,initial,[])
    started=time.perf_counter()
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','train_accuracy_augmented','samples','seconds',
            'train_eval_loss','train_eval_top1','val_eval_loss','val_eval_top1'])
        writer.writeheader()
        for epoch in range(1,options['epochs']+1):
            set_training_mode(model,options['bn_policy'])
            epoch_before=bn_snapshot(model);trace=[];tail=None
            total=correct=0;loss_sum=0.;epoch_start=time.perf_counter()
            for batch_index,(images,labels,indices) in enumerate(loader):
                last=batch_index==len(loader)-1
                tail_before=bn_snapshot(model) if last else None
                batch=dict(size=len(labels),indices=indices.tolist(),augmented_input_sha256=tensor_sha256(images),
                           class_counts={key:int((labels==i).sum()) for i,key in enumerate(keys)})
                trace.append(batch)
                images,labels=images.to(device),labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(images);loss=criterion(logits,labels)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(labels);correct+=(logits.argmax(1)==labels).sum().item();total+=len(labels)
                if last:
                    tail=dict(size=batch['size'],class_counts=batch['class_counts'],bn_delta=bn_delta(tail_before,bn_snapshot(model)),
                        ema_weight={name:(layer['momentum'] if layer['momentum'] is not None else 1/(layer['num_batches_tracked']+1))
                                    if options['bn_policy']=='update' else 0 for name,layer in tail_before['layers'].items()},
                        small_batch=bool(batch['size']<32),
                        caution='A small batch has the same fixed-momentum EMA weight as a full batch; causal contribution is unproven.')
            monitored=finish_epoch(epoch,epoch_before,trace,tail)
            val_metrics=monitored['validation'].get('metrics',{})
            row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy_augmented=correct/total,samples=total,seconds=time.perf_counter()-epoch_start,
                     train_eval_loss=monitored['train_eval']['loss'],train_eval_top1=monitored['train_eval']['top1'],
                     val_eval_loss=val_metrics.get('loss'),val_eval_top1=val_metrics.get('top1'))
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
    metadata=dict(format='oven-yolo-exploratory/v1',architecture='yolo11n-cls',class_keys=keys,class_list_sha256=class_digest(keys),
        preprocess=PREPROCESS,independent_validation=False,ultralytics_version='8.3.230',epoch=options['epochs'],config=config,
        torch_version=str(torch.__version__),device=str(device),samples=len(entries),seconds=time.perf_counter()-started)
    temporary=out/'last.tmp'
    torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata},temporary)
    temporary.replace(out/'last.pt')
    metadata['checkpoint_sha256']=digest(out/'last.pt')
    (out/'run_info.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'EXPLORATORY_TRAINING_COMPLETED','epoch':options['epochs'],'independent_validation':False,'checkpoint_sha256':metadata['checkpoint_sha256']}))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--pretrained',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    rows=json.loads(args.rows.read_text(encoding='utf-8'))
    keys=sorted({r['class_key'] for r in rows})
    active=check_rows(rows,keys)
    for row in active:
        if digest(input_path(row['relative_path']))!=row['sha256']:raise ValueError('Source image changed since audit')
    fit_exploratory([(input_path(r['relative_path']),r['class_key']) for r in active],keys,args.out,
                    args.pretrained,args.device,digest(args.rows))


if __name__=='__main__':main()
