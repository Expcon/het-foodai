"""Real Tk+ORT explicit catalog sequence; file-fed images, never hardware acceptance."""
import argparse,csv,json,sys,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import oven_capture_gui as gui
from model_catalog import MANUAL_LABEL
from inference_engine import read_image_unicode


def wait(root,done):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        root.update();root.update_idletasks()
        if done():return
        time.sleep(.02)
    raise TimeoutError('GUI model worker timed out')


def run(args):
    if args.out.exists():raise ValueError('Preserve previous evidence; use new output')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAIL',hardware_opened=False,frame_source='FILE-FED real image',checks={},models=[])
    root=None
    try:
        with patch.object(gui.DesktopApp,'refresh_cameras',lambda self:None),patch.object(gui.DesktopApp,'refresh_serials',lambda self:None):
            root=gui.RootClass();app=gui.DesktopApp(root,catalog_path=args.catalog)
        sequence=getattr(args,'sequence',['v2','R1','v2'])
        root.title(' / '.join(sequence)+' FILE-FED catalog smoke — NO CAMERA')
        assert app.model is None and app.model_profile.get()==MANUAL_LABEL and app.model_file.get()==''
        report['checks']['startup_does_not_load_or_change_default']=True
        identifiers={entry['id']:label for label,entry in app.model_profiles.items()}
        assert set(sequence)<=set(identifiers)
        frame=read_image_unicode(args.image,'oven_round2')
        for ident in sequence:
            label=identifiers[ident];app.model_profile.set(label);app.select_model_profile()
            assert app.model is None and not app.model_loading
            assert '未就绪' in app.model_state.get()
            app.load_model();wait(root,lambda:not app.model_loading)
            assert app.model is not None,app.model_state.get()
            assert app.model.contract['source_checkpoint_sha256']==app.model_profiles[label]['checkpoint_sha256']
            result=app.model.predict(frame)
            assert len(result['top3'])==3 and 'accuracy' not in json.dumps(result).lower()
            app.latest_frame=frame.copy();app.infer_current();held=app.model
            other=identifiers['v2' if ident=='R1' else 'R1']
            app.model_profile.set(other);app.select_model_profile()
            assert app.model is held and app.model_profile.get()==label
            wait(root,lambda:not app.inference_running)
            report['models'].append(dict(id=ident,checkpoint_sha256=app.model.contract['source_checkpoint_sha256'],top3=result['top3']))
        report['checks']['_'.join(sequence)+'_real_load_and_prediction']=True
        report['checks']['busy_switch_refused']=True
        app.latest_frame=None;app.infer_current();assert not app.inference_running
        report['checks']['absent_frame_refused']=True
        app.model_profile.set(MANUAL_LABEL);app.select_model_profile()
        app.model_file.set(str(args.out.parent/'missing.onnx'));app.load_model()
        wait(root,lambda:not app.model_loading)
        assert app.model is None and '加载失败' in app.model_state.get()
        report['checks']['missing_model_clears_ready']=True
        assert app.worker is None and app.serial_port is None
        report['status']='PASS'
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
    finally:
        if root is not None:root.destroy()
        with args.out.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=True));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['catalog','image','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sequence',nargs='+',default=['v2','R1','v2'])
    raise SystemExit(run(parser.parse_args()))
