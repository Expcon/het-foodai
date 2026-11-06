"""Export the real exploratory state_dict, then compare every supplied oven image.

No training, automatic downloads, labels-as-truth, or hardware access. Run export
with the project venv; only after PASS run mnn with the isolated edge venv.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from .model import (ROOT, PREPROCESS, configure_sdk, digest, input_path,
                    load_model, output_dir, transform)

ATOL = 1e-4


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def compare_probabilities(reference, actual):
    import numpy as np
    a, b = np.asarray(reference), np.asarray(actual)
    if a.ndim != 1 or a.shape != b.shape or a.size < 3:
        raise ValueError('Expected matching class vectors')
    if not all(np.isfinite(x).all() for x in (a,b)):
        raise ValueError('Expected finite probabilities')
    valid=all(x.min()>=0 and x.max()<=1 and np.isclose(x.sum(),1,atol=1e-5) for x in (a,b))
    ta, tb = np.argsort(-a,kind='stable')[:3], np.argsort(-b,kind='stable')[:3]
    error=np.abs(a.astype('float64')-b.astype('float64'))
    return dict(top1_agree=bool(ta[0]==tb[0]), top3_agree=bool(np.array_equal(ta,tb)),
                reference_top3=ta.tolist(), actual_top3=tb.tolist(), max_abs_error=float(error.max()),
                mean_abs_error=float(error.mean()), valid_probabilities=bool(valid),
                actual_probability_sum=float(b.sum()), **{'pass': bool(valid and np.array_equal(ta,tb) and error.max()<=ATOL)})


def make_contract(meta, checkpoint_sha, onnx_sha):
    return dict(format='het-oven-classifier/v1', class_keys=meta['class_keys'],
                classes=[dict(id=i,key=k) for i,k in enumerate(meta['class_keys'])],
                class_list_sha256=meta['class_list_sha256'], preprocess=PREPROCESS,
                input_shape=[1,3,224,224], output_shape=[1,len(meta['class_keys'])],
                output_kind='classification-probabilities', independent_validation=False,
                source_checkpoint_sha256=checkpoint_sha, models={'onnx':dict(sha256=onnx_sha)})


def summarize(records, versions):
    return dict(status='PASS' if records and all(r['pass'] for r in records) else 'FAIL',
                images=len(records), top1_agree=sum(r['top1_agree'] for r in records),
                ordered_top3_agree=sum(r['top3_agree'] for r in records),
                invalid_probability_vectors=sum(not r['valid_probabilities'] for r in records),
                max_abs_error=max((r['max_abs_error'] for r in records),default=None),
                acceptance={'max_absolute_probability_error':ATOL, 'require_all_ordered_top3':True},
                versions=versions, scope='format alignment on previously seen oven images, NOT accuracy or deployment acceptance')


def revoke_mnn(contract_path):
    """A failed recheck must never retain an earlier MNN authorization."""
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    if contract['models'].pop('mnn',None) is not None:
        write_json(contract_path,contract)
    return contract


def export_and_align(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image
    configure_sdk()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    out=output_dir(args.out)
    out.mkdir(parents=True,exist_ok=False)
    model,meta=load_model(args.checkpoint,'cpu',args.expect_sha256)
    if digest(args.rows)!=meta.get('config',{}).get('rows_sha256'):
        raise ValueError('Alignment requires the complete original rows bound to this checkpoint')

    class Probabilities(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network=network
        def forward(self, x):
            return self.network(x)[0]

    wrapper=Probabilities(model).eval()
    model_path=out/'oven9.onnx'
    # Explicit legacy exporter: static batch 1, unfused FP32, probability output.
    with torch.inference_mode():
        torch.onnx.export(wrapper,torch.zeros(1,3,224,224),str(model_path),
                          input_names=['images'],output_names=['probabilities'],
                          opset_version=17,dynamo=False,external_data=False)
    graph=onnx.load(model_path)
    onnx.checker.check_model(graph,full_check=True)
    onnx.helper.set_model_props(graph,{'het_contract_format':'het-oven-classifier/v1',
                                    'class_list_sha256':meta['class_list_sha256'],
                                    'preprocess':json.dumps(PREPROCESS,sort_keys=True)})
    onnx.save(graph,model_path)
    contract=make_contract(meta,digest(args.checkpoint),digest(model_path))
    # Class file is published only after all image checks succeed.
    write_json(out/'contract.pending.json',contract)
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    rows=json.loads(args.rows.read_text(encoding='utf-8'))
    if not rows: raise ValueError('No real image records')
    tensors=np.lib.format.open_memmap(out/'inputs.npy',mode='w+',dtype=np.float32,shape=(len(rows),3,224,224))
    reference=[]
    records=[]
    pre=transform(False)
    started=time.perf_counter()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            path=input_path(row['relative_path'])
            if digest(path)!=row['sha256']:raise ValueError('Original image SHA mismatch')
            with Image.open(path) as image: tensor=pre(image).unsqueeze(0)
            array=tensor.numpy()
            tensors[i]=array[0]
            p=wrapper(tensor).numpy()[0]
            q=session.run(['probabilities'],{'images':array})[0][0]
            record=compare_probabilities(p,q)
            record.update(image_id=row['image_id'],sha256=row['sha256'],reference_probabilities=p.tolist(),actual_probabilities=q.tolist())
            records.append(record)
            reference.append(p)
            if (i+1)%50==0: print(f'Compared {i+1}/{len(rows)}',flush=True)
    tensors.flush()
    del tensors
    np.save(out/'pytorch_probabilities.npy',np.asarray(reference),allow_pickle=False)
    versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','ultralytics','onnx','onnxruntime','numpy','Pillow']}
    summary=summarize(records,versions)
    summary.update(checkpoint_sha256=contract['source_checkpoint_sha256'],onnx_sha256=digest(model_path),
                   onnx_bytes=model_path.stat().st_size,rows_sha256=digest(args.rows),
                   input_tensors_sha256=digest(out/'inputs.npy'),reference_sha256=digest(out/'pytorch_probabilities.npy'),
                   elapsed_comparison_seconds=time.perf_counter()-started,input_shape=[1,3,224,224],output_shape=[1,9])
    write_json(out/'onnx_alignment.json',dict(summary=summary,images=records))
    if summary['status']=='PASS': write_json(out/'classes.json',contract)
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='PASS' else 2


def convert_mnn(args):
    out=output_dir(args.out)
    contract=revoke_mnn(out/'classes.json')
    from tools.edge_export.offline_guard import install
    install()
    import numpy as np
    import _tools
    import MNN
    report=json.loads((out/'onnx_alignment.json').read_text(encoding='utf-8'))
    summary=report['summary']
    if summary['status']!='PASS' or digest(out/'oven9.onnx')!=summary['onnx_sha256']:
        raise ValueError('ONNX must pass alignment first with the same bytes')
    for file,key in [('inputs.npy','input_tensors_sha256'),('pytorch_probabilities.npy','reference_sha256')]:
        if digest(out/file)!=summary[key]:raise ValueError('Alignment evidence hash mismatch')
    target=out/'oven9.mnn'
    if args.action=='mnn':
        if target.exists():raise ValueError('Refusing to overwrite prior MNN conversion; use check-mnn to audit existing bytes')
        _tools.mnnconvert(['mnnconvert','-f','ONNX','--modelFile',str(out/'oven9.onnx'),
                          '--MNNModel',str(target),'--bizCode','HeT-exploratory-oven9'])
    if not target.is_file():raise ValueError('Converter did not create MNN')
    runtime=MNN.nn.create_runtime_manager(({'backend':'CPU','precision':'high','numThread':4},))
    module=MNN.nn.load_module_from_file(str(target),[],[],runtime_manager=runtime,rearrange=True)
    inputs=np.load(out/'inputs.npy',mmap_mode='r',allow_pickle=False)
    ref=np.load(out/'pytorch_probabilities.npy',allow_pickle=False)
    records=[]
    for i,array in enumerate(inputs):
        x=np.ascontiguousarray(array[None])
        y=np.asarray(module.onForward([MNN.expr.const(x,x.shape,MNN.expr.NCHW)])[0].read()).copy()[0]
        result=compare_probabilities(ref[i],y)
        result.update(image_id=report['images'][i]['image_id'],sha256=report['images'][i]['sha256'],
                      reference_probabilities=ref[i].tolist(),actual_probabilities=y.tolist())
        records.append(result)
        if (i+1)%50==0:print(f'MNN compared {i+1}/{len(inputs)}',flush=True)
    result=summarize(records,{k:importlib.metadata.version(k) for k in ['MNN','numpy']})
    result.update(onnx_sha256=digest(out/'oven9.onnx'),mnn_sha256=digest(target),mnn_bytes=target.stat().st_size,
                  precision='FP32 CPU high; no INT8 quantization; no ARM hardware test')
    write_json(out/'mnn_alignment.json',dict(summary=result,images=records))
    if result['status']=='PASS':
        contract['models']['mnn']={'sha256':digest(target)}
        write_json(out/'classes.json',contract)
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','mnn','check-mnn'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expect-sha256')
    p.add_argument('--rows',type=Path)
    args=p.parse_args()
    if args.action=='export' and (not args.checkpoint or not args.rows or not args.expect_sha256):
        p.error('export requires --checkpoint, --expect-sha256, --rows')
    return export_and_align(args) if args.action=='export' else convert_mnn(args)


if __name__=='__main__':
    raise SystemExit(main())
