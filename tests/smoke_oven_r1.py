"""Actual YOLO CUDA SYNTHETIC R0/R1 input controls and safe R1 recovery."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset
from tools.oven_training.model import configure_sdk,digest,new_model,output_dir,PRETRAINED_SHA
from tools.oven_training.round2 import V2_KEYS
from tools.oven_training.train import fit_exploratory
from tools.oven_training.observation import load_snapshot,restore_training,state_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True)
    args=parser.parse_args()
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Pinned official pretrained required')
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    generator=np.random.default_rng(20261111);entries=[]
    for i in range(65):
        path=out/f'SYNTHETIC_{i:03d}.png';Image.fromarray(generator.integers(0,256,(180,320,3),dtype=np.uint8)).save(path)
        entries.append((path,V2_KEYS[i%12]))
    for name,recipe,observation in [('A',{'epochs':2},None),('R0',{'epochs':2},{'reference':out/'A','tail_epochs':(1,2)}),
                                   ('R1',{'epochs':2,'lr':.0003},{'reference':out/'R0','comparison':'lr_only','tail_epochs':(1,2)})]:
        fit_exploratory(entries,V2_KEYS,out/name,args.pretrained,'cuda:0','a'*64,
            provenance={'scope':'SYNTHETIC SOFTWARE ONLY; arbitrary labels; no food accuracy'},recipe=recipe,observation=observation)
    obs=out/'R1/observations';gate=json.loads((obs/'gate.json').read_text())
    assert gate['status']=='PASS' and gate['comparison_mode']=='lr_only' and gate['only_config_difference']=='lr'
    assert gate['all_reference_batch_traces_equal'] and not gate['all_reference_metrics_and_batches_equal']
    for e in [1,2]:
        a=json.loads((out/f'R0/monitor/epoch_{e:03d}.json').read_text());b=json.loads((out/f'R1/monitor/epoch_{e:03d}.json').read_text())
        assert a['batch_trace']==b['batch_trace']
    records={Path(r['file']).name:r for r in [json.loads(s) for s in (obs/'checkpoints.jsonl').read_text().splitlines()]}
    def read(name):
        r=records[name];return load_snapshot(obs/r['file'],r['sha256'])
    pre=read('epoch_002_tail_pre.pt');post=read('epoch_002_tail_post.pt')
    configure_sdk(out/'recovery-sdk');model=new_model(V2_KEYS).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0005)
    empty=TensorDataset(torch.empty(0,3,224,224),torch.empty(0,dtype=torch.long))
    loaders={name:DataLoader(empty,generator=torch.Generator()) for name in pre['loader_generators']}
    restore_training(pre,model,opt,loaders,pre['binding'])
    images,labels,indices=pre['pending'];opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(images.cuda()),labels.cuda());loss.backward();opt.step()
    assert state_digest(model.state_dict())==post['model_state_sha256']
    for i,values in post['optimizer_state']['state'].items():
        for key,value in values.items():assert torch.equal(opt.state_dict()['state'][i][key].cpu(),value)
    result=dict(status='PASS',scope='SYNTHETIC SOFTWARE ONLY',device=torch.cuda.get_device_name(),samples=65,epochs=2,
        lr_R0=.001,lr_R1=.0003,initial_state_equal=True,all6_batch_indices_and_augmented_inputs_equal=True,
        only_active_config_difference='lr',checkpoints=len(records),safe_pending_recovery_model_and_optimizer_exact=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
