"""Real-checkpoint field CLI check; only malformed inputs are SYNTHETIC."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expect-sha256',required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():raise ValueError('Fresh ignored runs directory required')
    out.mkdir(parents=True)
    jpeg=out/'DERIVED-from-real-source.jpg'
    with Image.open(args.image) as image:image.convert('RGB').save(jpeg,quality=95)
    corrupt=out/'SYNTHETIC-corrupt.png';corrupt.write_bytes(b'SYNTHETIC not an image')
    bad_model=out/'SYNTHETIC-corrupt.pt';bad_model.write_bytes(b'SYNTHETIC not a checkpoint')
    bad_sha=hashlib.sha256(bad_model.read_bytes()).hexdigest()
    prefix=[sys.executable,'-m','tools.oven_training.predict']
    common=['--checkpoint',str(args.checkpoint),'--expect-sha256',args.expect_sha256,'--device','cpu']
    cases=[('real_png_and_derived_jpg',common+['--images',str(args.image),str(jpeg)],0),
        ('missing_image',common+['--images',str(out/'MISSING.png')],2),
        ('corrupt_image',common+['--images',str(corrupt)],2),
        ('no_image_argument',common,2),
        ('invalid_batch',common+['--images',str(args.image),'--batch-size','0'],2),
        ('missing_model',['--checkpoint',str(out/'MISSING.pt'),'--expect-sha256',args.expect_sha256,'--images',str(args.image)],2),
        ('corrupt_model',['--checkpoint',str(bad_model),'--expect-sha256',bad_sha,'--images',str(args.image)],2)]
    results=[]
    for name,options,expected in cases:
        result=subprocess.run([*prefix,*options,'--out',str(out/name)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (out/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        assert result.returncode==expected,(name,result.returncode,result.stderr)
        if expected==2:assert not (out/name/'predictions.json').exists()
        results.append(dict(case=name,exit_code=result.returncode,expected_exit_code=expected))
    predictions=json.loads((out/'real_png_and_derived_jpg/predictions.json').read_text(encoding='utf-8'))
    assert predictions['image_count']==2 and all(len(r['top3'])==3 for r in predictions['predictions'])
    proof=dict(state='PASS',real_checkpoint=True,real_png=True,jpeg_is_reencoding_not_independent_sample=True,
        malformed_inputs_synthetic=True,camera_opened=False,cases=results)
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
