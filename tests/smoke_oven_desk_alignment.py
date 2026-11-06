"""Real-image Desk-runtime parity, no camera and no accuracy calculation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools/oven_desk'))


def main():
    import numpy as np
    from inference_engine import ModelEngine, read_image_unicode, prepare_input
    from tools.oven_training.export_desk import compare_probabilities, summarize, write_json
    from tools.oven_training.model import digest, input_path
    import importlib.metadata
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True)
    args=p.parse_args()
    out=args.bundle/'desk_alignment.json'
    if out.exists():raise ValueError('Use a new output bundle; preserve prior results')
    evidence=json.loads((args.bundle/'onnx_alignment.json').read_text(encoding='utf-8'))
    summary=evidence['summary']
    if digest(args.rows)!=summary['rows_sha256']:raise ValueError('Rows changed')
    rows=json.loads(args.rows.read_text(encoding='utf-8'))
    for name,key in [('inputs.npy','input_tensors_sha256'),('pytorch_probabilities.npy','reference_sha256')]:
        if digest(args.bundle/name)!=summary[key]:raise ValueError('Reference evidence changed')
    inputs=np.load(args.bundle/'inputs.npy',mmap_mode='r',allow_pickle=False)
    reference=np.load(args.bundle/'pytorch_probabilities.npy',allow_pickle=False)
    engine=ModelEngine(args.bundle/'oven9.onnx',args.bundle/'classes.json','oven_round1',224)
    records=[]
    for i,row in enumerate(rows):
        path=input_path(row['relative_path'])
        if digest(path)!=row['sha256']:raise ValueError('Original image changed')
        frame=read_image_unicode(path,mode='oven_round1')
        x,_=prepare_input(frame,'oven_round1',224,224)
        if not np.array_equal(x[0],inputs[i]):raise ValueError(f'Desk preprocessing differs: {row["image_id"]}')
        raw=engine.session.run(None,{engine.input_name:x})[0][0]
        result=engine.predict(frame)
        record=compare_probabilities(reference[i],raw)
        assert [v['id'] for v in result['top3']]==record['actual_top3']
        assert all(abs(v['score']-float(raw[v['id']]))<1e-7 for v in result['top3'])
        assert 'accuracy' not in result
        record.update(image_id=row['image_id'],sha256=row['sha256'],actual_probabilities=raw.tolist())
        records.append(record)
        if (i+1)%50==0:print(f'Desk compared {i+1}/{len(rows)}',flush=True)
    result=summarize(records,{k:importlib.metadata.version(k) for k in ['numpy','Pillow','onnxruntime','opencv-python']})
    result.update(preprocessing_tensors_exact=len(rows),onnx_sha256=digest(args.bundle/'oven9.onnx'),
                  classes_sha256=digest(args.bundle/'classes.json'))
    write_json(out,dict(summary=result,images=records))
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
