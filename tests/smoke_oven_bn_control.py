"""Two short SYNTHETIC GPU fits, never food metrics or a deployable model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools.oven_training.model import ROOT, PRETRAINED_SHA, digest, output_dir
from tools.oven_training.round2 import V2_KEYS
from tools.oven_training.train import fit_exploratory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Pinned pretrained required; no download')
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    protected=[ROOT/'configs/classes.json',ROOT/'configs/classes.draft.json',ROOT/'data/manifest.csv',
        ROOT/'data/processed/selected_dataset_v1.1.0/manifest.csv',args.pretrained,
        ROOT/'runs/oven-first-round-20261106/yolo11n-fixed30/last.pt',ROOT/'runs/oven-desk-20261106/oven9.onnx',
        ROOT/'runs/oven-desk-20261106/classes.json',ROOT/'runs/oven-round2-20261107/yolo11n-12cls-fixed30/last.pt',
        ROOT/'runs/oven-round2-20261107/onnx/oven12.onnx',ROOT/'runs/oven-round2-20261107/onnx/classes.json',
        ROOT/'runs/oven-round2-20261107/audit/combined_rows.json']
    before={str(path):digest(path) for path in protected if path.exists()}
    rng=np.random.default_rng(20261111);entries=[]
    for index in range(65):
        path=out/f'SYNTHETIC_{index:03d}.png'
        Image.fromarray(rng.integers(0,256,size=(180,320,3),dtype=np.uint8)).save(path)
        entries.append((path,V2_KEYS[index%12]))
    source_hashes={str(path):digest(path) for path,_ in entries}
    rows_sha=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    for policy in ['update','frozen_stats']:
        fit_exploratory(entries,V2_KEYS,out/policy,args.pretrained,args.device,rows_sha,
                        provenance={'scope':'SYNTHETIC SOFTWARE SMOKE ONLY; random pixels and arbitrary labels; not food accuracy'},
                        recipe={'epochs':2,'bn_policy':policy})
    def read(policy,name):return json.loads((out/policy/'monitor'/name).read_text())
    a=read('update','initialization.json');b=read('frozen_stats','initialization.json')
    assert a['model_state_sha256']==b['model_state_sha256'],'Different initial parameters or BN buffers'
    assert a['trainable_parameters']==b['trainable_parameters']
    for epoch in [1,2]:
        normal=read('update',f'epoch_{epoch:03d}.json');frozen=read('frozen_stats',f'epoch_{epoch:03d}.json')
        assert normal['batch_trace']==frozen['batch_trace'],'Different shuffled or augmented inputs'
        assert normal['batch_trace']['sizes']==[32,32,1]
        for record in [normal,frozen]:
            assert record['train_eval']['samples']==65 and record['train_eval']['bn_unchanged']
            assert record['validation']['status']=='BLOCKED'
        assert all(layer['counter_delta']==3 for layer in normal['bn_update']['layers'].values())
        assert frozen['bn_update']['running_unchanged']
        assert frozen['bn']['running_sha256']==b['bn']['running_sha256']
    # New synthetic checkpoints must retain the same safe loader/inference contract.
    from tools.oven_training.model import load_model
    for policy in ['update','frozen_stats']:
        model,metadata=load_model(out/policy/'last.pt',args.device)
        assert metadata['class_keys']==V2_KEYS and metadata['epoch']==2
        del model
    assert all(digest(Path(path))==sha for path,sha in before.items())
    assert all(digest(Path(path))==sha for path,sha in source_hashes.items())
    result=dict(status='PASS',scope='SYNTHETIC SOFTWARE SMOKE ONLY; no food accuracy',device=args.device,
        samples=65,epochs_per_arm=2,batch_sizes=[32,32,1],bn_layers=len(a['bn']['layers']),
        initial_state_equal=True,trainable_parameter_count_equal=True,all_augmented_batch_inputs_equal=True,
        normal_counter_increase_per_epoch=3,frozen_running_buffers_unchanged=True,fixed_eval_bn_unchanged=True,
        validation_status='BLOCKED',new_synthetic_checkpoints_safely_loaded=True,protected_files_unchanged=len(before))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
