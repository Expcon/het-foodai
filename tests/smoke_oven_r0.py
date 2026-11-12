"""Two-epoch SYNTHETIC CUDA observation transparency and real-model recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from tools.oven_training.model import ROOT, PRETRAINED_SHA, configure_sdk, digest, new_model, output_dir
from tools.oven_training.round2 import V2_KEYS
from tools.oven_training.train import fit_exploratory
from tools.oven_training.observation import load_snapshot, restore_training, resume_batches, state_digest
from tools.oven_training.monitor import IndexedImages
from torch.utils.data import DataLoader, Dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True)
    args=parser.parse_args()
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Pinned pretrained SHA required')
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    generator=np.random.default_rng(20261111);entries=[]
    for i in range(65):
        path=out/f'SYNTHETIC_{i:03d}.png'
        Image.fromarray(generator.integers(0,256,(180,320,3),dtype=np.uint8)).save(path)
        entries.append((path,V2_KEYS[i%12]))
    for name,observation in [('plain',None),('observed',{'reference':out/'plain','tail_epochs':(1,2)})]:
        fit_exploratory(entries,V2_KEYS,out/name,args.pretrained,'cuda:0','a'*64,
            provenance={'scope':'SYNTHETIC SOFTWARE ONLY; arbitrary labels; not food accuracy'},
            recipe={'epochs':2},observation=observation)
    observation_root=out/'observed/observations'
    records=[json.loads(s) for s in (observation_root/'checkpoints.jsonl').read_text().splitlines()]
    assert len(records)==7
    # Replay the actual stored final pending tensor, without drawing any new augmentation.
    pre=next(r for r in records if r['file'].endswith('epoch_002_tail_pre.pt'))
    post=next(r for r in records if r['file'].endswith('epoch_002_tail_post.pt'))
    payload=load_snapshot(observation_root/pre['file'],pre['sha256'])
    expected=load_snapshot(observation_root/post['file'],post['sha256'])
    configure_sdk(out/'recovery-sdk')
    model=new_model(V2_KEYS).to('cuda:0');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    # Generator restoration needs the real loader identities, not their dataset for a pending step.
    class Unused(Dataset):
        def __len__(self):return 65
        def __getitem__(self,index):raise AssertionError('Pending tensor must not be augmented again')
    loaders={key:DataLoader(Unused(),batch_size=32,generator=torch.Generator()) for key in payload['loader_generators']}
    restore_training(payload,model,optimizer,loaders,payload['binding'])
    images,labels,indices=payload['pending']
    optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(images.cuda()),labels.cuda());loss.backward();optimizer.step()
    assert state_digest(model.state_dict())==expected['model_state_sha256']
    actual_optimizer=optimizer.state_dict()
    assert actual_optimizer['param_groups']==expected['optimizer_state']['param_groups']
    for i,values in expected['optimizer_state']['state'].items():
        for key,value in values.items():assert torch.equal(actual_optimizer['state'][i][key].cpu(),value),(i,key)
    result=dict(status='PASS',scope='SYNTHETIC SOFTWARE ONLY',device=torch.cuda.get_device_name(),samples=65,epochs=2,
        observation_transparent=True,seven_checkpoints_safely_read=True,pending_tail_actual_model_and_optimizer_exact=True,
        tail_samples=len(labels),tail_loss=float(loss.detach()),checkpoint_bytes=[r['bytes'] for r in records])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
