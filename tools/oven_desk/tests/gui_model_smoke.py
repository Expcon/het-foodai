"""Real-model Tk smoke with FILE-FED frames, never a physical-camera test.

python tests/gui_model_smoke.py --model oven9.onnx --classes classes.json
    --image real-photo.png --out <ignored-runs>/source-gui-smoke.json
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oven_capture_gui as gui
from inference_engine import read_image_unicode, file_sha256


def pump_until(root, predicate, timeout=30):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        root.update_idletasks()
        root.update()
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('Timed out waiting for GUI worker/event completion')


def run(args):
    target = args.out.resolve()
    if target.exists(): raise ValueError('Use a new report path; existing evidence is preserved')
    artifacts = target.parent/(target.stem+'-artifacts')
    artifacts.mkdir(parents=True, exist_ok=False)
    gallery = artifacts/'FILE-FED-derived-gallery'
    gallery.mkdir()
    empty = artifacts/'empty-gallery'
    empty.mkdir()
    frame = read_image_unicode(args.image, mode='oven_round1')
    rgb = Image.fromarray(np.ascontiguousarray(frame[:,:,::-1]))
    rgb.save(gallery/'real-file-decoded.png')
    rgb.save(gallery/'real-file-derived.jpg', quality=95)
    (gallery/'corrupt-test-fixture.jpg').write_bytes(b'SYNTHETIC corrupt-image negative test')
    report = dict(status='FAIL', camera_opened=False, serial_opened=False,
                  frame_source='FILE-FED real image; not a physical camera',
                  independent_validation=False, source_image_sha256=file_sha256(args.image),
                  model_sha256=file_sha256(args.model), checks={}, dialogs=[], logs=[])
    root = None
    try:
        # The constructor schedules enumeration; suppress only that device interaction.
        with patch.object(gui.DesktopApp, 'refresh_cameras', lambda self: None):
            root = gui.RootClass()
            app = gui.DesktopApp(root)
        root.title(root.title()+' · FILE-FED smoke / NO CAMERA')
        panel = app.model_tab.master.master
        panel.master.select(panel)
        app.output.set(str(artifacts/'csv-output'))
        app.model_file.set(str(args.model.resolve()))
        app.label_file.set(str(args.classes.resolve()))
        app.model_task.set(gui.OVEN_MODE_LABEL)
        app.network_size.set('224')
        original_log = app._log
        def record_log(text):
            report['logs'].append(text)
            original_log(text)
        app._log = record_log
        def record_dialog(title, text):
            report['dialogs'].append(dict(title=title, text=text))
        with patch.object(gui.messagebox, 'showwarning', record_dialog), patch.object(gui.messagebox, 'showinfo', record_dialog):
            app.infer_current()
            assert not app.inference_running and app.latest_frame is None
            report['checks']['absent_camera_frame_refused'] = True
            app.load_model()
            assert app.model is None and app.model_loading
            pump_until(root, lambda: not app.model_loading)
            assert app.model is not None, app.model_state.get()
            report['checks']['real_model_loaded_via_gui'] = True
            app.camera_status.set('FILE-FED real image · NO physical camera')
            app.latest_frame = frame.copy()
            app._log('FILE-FED smoke：下一条“摄像头当前帧”来自真实图片文件，不是摄像头。')
            app.infer_current()
            engine = app.model
            app.load_model()
            assert app.model is engine and not app.model_loading
            pump_until(root, lambda: not app.inference_running)
            assert any('真实模型预测' in item for item in report['logs'])
            assert not any('预测失败' in item for item in report['logs'])
            report['checks']['file_fed_current_frame_prediction'] = True
            report['checks']['model_switch_refused_while_busy'] = True
            with patch.object(gui.filedialog, 'askdirectory', return_value=str(gallery)):
                app.test_gallery()
            pump_until(root, lambda: not app.inference_running)
            csv_files = list((artifacts/'csv-output').glob('*.csv'))
            assert len(csv_files) == 1
            with csv_files[0].open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 3 and sum(row['status']=='OK' for row in rows) == 2, rows
            assert 'accuracy' not in json.dumps(rows).lower()
            report['checks']['png_jpeg_predictions_and_corrupt_file_error'] = True
            report['gallery_csv'] = str(csv_files[0])
            report['gallery_execution'] = dict(total=3, succeeded=2, failed=1)
            with patch.object(gui.filedialog, 'askdirectory', return_value=str(empty)):
                app.test_gallery()
            assert not app.inference_running
            assert len(list((artifacts/'csv-output').glob('*.csv'))) == 1
            assert any('没有 JPG/PNG' in item['text'] for item in report['dialogs'])
            report['checks']['empty_folder_refused'] = True
            app.model_file.set(str(artifacts/'missing.onnx'))
            app.load_model()
            assert app.model is None
            pump_until(root, lambda: not app.model_loading)
            assert app.model is None and '加载失败' in app.model_state.get()
            report['checks']['failed_reload_clears_old_model'] = True
            assert app.worker is None and app.serial_port is None
            report['checks']['hardware_not_opened'] = True
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if root is not None:
            # No user settings are changed by this isolated smoke.
            root.destroy()
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status':report['status'], 'report':str(target), 'checks':report['checks']}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--classes', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__': raise SystemExit(main())
