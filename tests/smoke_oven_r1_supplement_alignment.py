"""Real diagnostic-image parity through the Desk file pipeline. No accuracy metric."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools/oven_desk'))


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from inference_engine import ModelEngine,prepare_input,read_image_unicode
    from tools.oven_training.model import configure_sdk,digest,load_model,output_dir,transform
    from tools.oven_training.monitor import model_state_sha256
    from tools.oven_training.export_desk import compare_probabilities,summarize
    from tools.oven_training.supplement import R1_SHA
    from tools.oven_training.round2_audit import safe_child
    from tools.oven_training.observation import write_json_new
    out=output_dir(args.out)
    if out.exists() or out.is_relative_to(args.source_root.resolve()):raise ValueError('Use new evidence path outside original images')
    plan=json.loads(args.plan.read_text('utf-8'))
    if digest(args.inventory)!=plan['inventory_sha256']:raise ValueError('Inventory differs from frozen diagnostic plan')
    rows=json.loads(args.inventory.read_text('utf-8'))
    if not rows or any(r['route']!='diagnostic_only' for r in rows):raise ValueError('Only exposed diagnostic rows; sealed test prohibited')
    report=json.loads((args.bundle/'onnx_alignment.json').read_text('utf-8'))['summary']
    if report['status']!='PASS' or digest(args.bundle/'oven12.onnx')!=report['onnx_sha256'] or report['checkpoint_sha256']!=R1_SHA:
        raise ValueError('Original export evidence/model/checkpoint mismatch')
    configure_sdk(out.parent/'alignment-sdk');torch.set_num_threads(4)
    model,meta=load_model(args.checkpoint,'cpu',R1_SHA);before=model_state_sha256(model)
    engine=ModelEngine(args.bundle/'oven12.onnx',args.bundle/'classes.json','oven_round2',224)
    if engine.contract['source_checkpoint_sha256']!=R1_SHA or engine.labels!=meta['class_keys']:
        raise ValueError('Desk ordered classes/checkpoint mismatch')
    bound={str(path):digest(path) for path in [args.checkpoint,args.inventory,args.plan,args.bundle/'oven12.onnx',args.bundle/'classes.json']}
    pre=transform(False);results=[]
    with torch.inference_mode():
        for row in rows:
            path=safe_child(args.source_root,row['relative_path'])
            if digest(path)!=row['sha256']:raise ValueError('Original source SHA changed')
            with Image.open(path) as image:tensor=pre(image).unsqueeze(0)
            frame=read_image_unicode(path,'oven_round2');actual_input,_=prepare_input(frame,'oven_round2',224,224)
            same=np.array_equal(tensor.numpy(),actual_input)
            p=model(tensor)[0].numpy()[0];result=engine.predict_file(path)
            q=np.asarray(result['probabilities'],dtype=np.float32)
            comparison=compare_probabilities(p,q)
            comparison['pass']=comparison['pass'] and same and [r['id'] for r in result['top3']]==comparison['actual_top3']
            comparison.update(image_id=row['image_id'],relative_path=row['relative_path'],sha256=row['sha256'],
                candidate_class_key=row['candidate_class_key'],preprocessing_exact=same,
                input_tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                reference_probabilities=p.tolist(),actual_probabilities=q.tolist(),
                pytorch_top3=[dict(class_id=i,class_key=engine.labels[i],score=float(p[i])) for i in comparison['reference_top3']],
                desk_top3=result['top3'])
            if digest(path)!=row['sha256']:raise ValueError('Source changed during inference')
            results.append(comparison)
    if model_state_sha256(model)!=before:raise ValueError('Frozen model state changed')
    for path,sha in bound.items():
        if digest(path)!=sha:raise ValueError('Bound inputs changed')
    summary=summarize(results,dict(torch=torch.__version__,numpy=np.__version__))
    summary.update(preprocessing_exact=sum(r['preprocessing_exact'] for r in results),checkpoint_sha256=R1_SHA,
        onnx_sha256=report['onnx_sha256'],model_state_unchanged=True,scope=f'{len(rows)} exposed diagnostic images; format alignment only, NOT independent accuracy')
    write_json_new(out,dict(summary=summary,images=results));print(json.dumps(summary))
    return 0 if summary['status']=='PASS' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','checkpoint','inventory','plan','source-root','out']:parser.add_argument('--'+key,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
