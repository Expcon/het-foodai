"""Fixed-epoch, train-only exploratory YOLO fit. No automatic validation fallback."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

from .model import ROOT,PREPROCESS,class_digest,check_rows,configure_sdk,digest,input_path,new_model,output_dir,frame_image,transform


def fit_exploratory(entries, keys, out, pretrained, device, rows_sha256, provenance=None):
    """Shared fixed recipe; caller validates hashes, labels and train-only groups."""
    out=output_dir(out)
    if out.exists():raise ValueError('Use a new run directory; never overwrite a training run')
    if not entries or any(key not in keys for _,key in entries):raise ValueError('Explicit known training labels required')
    class_digest(keys)
    config=dict(epochs=30,batch_size=32,seed=42,optimizer='AdamW',lr=.001,weight_decay=.0005,
        amp=False,workers=0,selection='final_epoch_only',validation=False,independent_validation=False,
        preprocessing=PREPROCESS,augmentation='horizontal_flip_p0.5;color_jitter_brightness_contrast_saturation_0.1',
        rows_sha256=rows_sha256,pretrained_sha256=digest(pretrained),class_keys=keys,
        provenance=provenance or {})
    out.mkdir(parents=True)
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
    loader=DataLoader(Images(),batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    model=new_model(keys,pretrained).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    criterion=torch.nn.CrossEntropyLoss()
    started=time.perf_counter()
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','train_accuracy_augmented','samples','seconds'])
        writer.writeheader()
        for epoch in range(1,31):
            model.train();total=correct=0;loss_sum=0.;epoch_start=time.perf_counter()
            for images,labels in loader:
                images,labels=images.to(device),labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(images);loss=criterion(logits,labels)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(labels);correct+=(logits.argmax(1)==labels).sum().item();total+=len(labels)
            row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy_augmented=correct/total,samples=total,seconds=time.perf_counter()-epoch_start)
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
    metadata=dict(format='oven-yolo-exploratory/v1',architecture='yolo11n-cls',class_keys=keys,class_list_sha256=class_digest(keys),
        preprocess=PREPROCESS,independent_validation=False,ultralytics_version='8.3.230',epoch=30,config=config,
        torch_version=str(torch.__version__),device=str(device),samples=len(entries),seconds=time.perf_counter()-started)
    temporary=out/'last.tmp'
    torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata},temporary)
    temporary.replace(out/'last.pt')
    metadata['checkpoint_sha256']=digest(out/'last.pt')
    (out/'run_info.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'EXPLORATORY_TRAINING_COMPLETED','epoch':30,'independent_validation':False,'checkpoint_sha256':metadata['checkpoint_sha256']}))
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
