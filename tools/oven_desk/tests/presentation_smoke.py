"""Real models/files through Tk; layout font-scale probes, never camera acceptance."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import oven_capture_gui as gui
from inference_engine import file_sha256,read_image_unicode
from model_catalog import MANUAL_LABEL
from recognition_session import frame_sha256


def wait(root,done):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        root.update();root.update_idletasks()
        if done():return
        time.sleep(.01)
    raise TimeoutError('GUI completion not received')


def create(catalog,scale=None,size='1366x768'):
    with patch.object(gui.DesktopApp,'refresh_cameras',lambda self:None),patch.object(gui.DesktopApp,'refresh_serials',lambda self:None):
        root=gui.RootClass()
        if scale is not None:root.tk.call('tk','scaling',scale)
        app=gui.DesktopApp(root,catalog_path=catalog)
    root.geometry(size);root.update()
    return root,app


def close(root):
    for item in root.tk.call('after','info'):root.after_cancel(item)
    root.destroy()


def run(args):
    if args.out.exists():raise ValueError('Use a new evidence path')
    artifacts=args.out.parent/(args.out.stem+'-artifacts');artifacts.mkdir(parents=True,exist_ok=False)
    baseline=json.loads(args.baseline.read_text('utf-8'))
    images=[Path(path) for path in baseline['images']]
    report=dict(status='FAIL',camera_opened=False,synthetic_model=False,independent_validation=False,
                comparisons=[],checks={},layout=[],dialogs=[])
    root=None
    try:
        root,app=create(args.catalog)
        labels={entry['id']:name for name,entry in app.model_profiles.items()}
        for ident in ['v2','R1','R2','v2']:
            app.model_profile.set(labels[ident]);app.select_model_profile()
            assert app.model is None
            app.load_model();wait(root,lambda:not app.model_loading)
            assert app.model is not None,app.model_state.get()
            for index,image in enumerate(images):
                app.open_image_file(image);assert app.recognition.record is None
                app.infer_current();sequence=app.recognition.sequence;app.infer_current()
                assert app.recognition.sequence==sequence
                wait(root,lambda:not app.inference_running)
                record=app.recognition.record;expected=baseline['models'][ident]['records'][index]
                assert record['prediction']==expected['result']
                assert record['frame_sha256']==frame_sha256(read_image_unicode(image,'oven_round2'))
                assert record['source']['original_sha256']==expected['source_sha256']==file_sha256(image)
                assert record['model']['model_sha256']==baseline['models'][ident]['model_sha256']
                report['comparisons'].append(dict(model=ident,source_sha256=expected['source_sha256'],
                    exact_baseline_output=True,top3=record['prediction']['top3'],latency_ms=record['latency_ms']))
        report['checks']['v2_R1_R2_v2_real_files_exact_baseline']=True
        report['checks']['duplicate_submission_refused']=True
        app.infer_current();app.open_image_file(images[0]);app.open_image_file(images[1])
        wait(root,lambda:not app.inference_running)
        assert app.recognition.record is None
        report['checks']['real_async_A_B_A_stale_completion_rejected']=True
        app.infer_current();wait(root,lambda:not app.inference_running)
        saved=app.recognition.save_result(artifacts/'saved-results')
        metadata=json.loads((saved/'result.json').read_text('utf-8'))
        assert metadata['saved_png_sha256']==file_sha256(saved/'frame.png')
        assert frame_sha256(read_image_unicode(saved/'frame.png','oven_round2'))==metadata['frame_sha256']
        assert metadata['prediction']==app.recognition.record['prediction']
        report['checks']['saved_exact_frame_model_and_source_identity']=True
        report['saved_result']=str(saved)
        # A real runtime is reused only for layout probes; every result/timer is measured anew.
        engine=app.model;identity=dict(app.loaded_identity)
        corrupt=artifacts/'corrupt-negative.png';corrupt.write_bytes(b'SYNTHETIC damaged-file negative fixture')
        app.open_image_file(corrupt)
        assert app.recognition.record is None and app._recognition_frame() is None
        assert '图片' in app.presentation.feedback.get()
        app.use_camera_source();app.infer_current();assert not app.inference_running
        report['checks']['corrupt_import_and_no_frame_clear_previous_success']=True
        with patch.object(gui.messagebox,'showwarning',lambda *a:report['dialogs'].append(a)):
            app.model_profile.set(MANUAL_LABEL);app.select_model_profile()
            good_model=app.model_file.get();good_classes=app.label_file.get()
            # Manual profile may retain the previous filenames by design.
            if not good_model:good_model=str(args.catalog.parent/'models/v2/oven12.onnx')
            invalid=artifacts/'invalid-contract.json';invalid.write_text('{"bad":"SYNTHETIC negative fixture"}',encoding='utf-8')
            app.model_file.set(good_model);app.label_file.set(str(invalid));app.load_model()
            wait(root,lambda:not app.model_loading)
            assert app.model is None and '加载失败' in app.model_state.get()
            report['checks']['wrong_contract_rejected']=True
            app.model_file.set(str(artifacts/'missing.onnx'));app.label_file.set(good_classes)
            app.load_model();wait(root,lambda:not app.model_loading)
            assert app.model is None and '加载失败' in app.model_state.get()
            report['checks']['missing_model_rejected']=True
        assert app.worker is None and app.serial_port is None
        close(root);root=None
        for size in ['1366x768','1920x1080']:
            for dpi in [96,120,144]:
                root,app=create(args.catalog,dpi/72,size)
                app.model=engine;app.loaded_identity=identity
                app.open_image_file(images[0]);app.infer_current();wait(root,lambda:not app.inference_running)
                root.update()
                ui=app.presentation
                widgets=[ui.recognize_button,ui.save_image_button,ui.save_result_button,ui.feedback_label,
                         ui.name_label,ui.scope_label,ui.source_label,ui.model_label,ui.result_thumbnail]
                for widget in widgets:
                    assert widget.winfo_ismapped(),str(widget)
                    assert widget.winfo_rooty()+widget.winfo_height()<=root.winfo_rooty()+root.winfo_height(),str(widget)
                    assert widget.winfo_rootx()+widget.winfo_width()<=root.winfo_rootx()+root.winfo_width(),str(widget)
                report['layout'].append(dict(client=size,tk_font_dpi=dpi,status='PASS',
                    actual_client=[root.winfo_width(),root.winfo_height()],scope='software font-scale probe, not OS display scaling change'))
                close(root);root=None
        report['status']='PASS'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        import traceback
        report['traceback']=traceback.format_exc()
    finally:
        if root is not None:close(root)
        with args.out.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=True));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['catalog','baseline','out']:parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
