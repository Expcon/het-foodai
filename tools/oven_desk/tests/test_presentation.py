"""SYNTHETIC Tk lifecycle tests; no real model/camera acceptance claims."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import oven_capture_gui as gui


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='SYNTHETIC-presentation-');self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.frame=np.full((90,160,3),114,dtype=np.uint8)
        self.image=self.folder/'A.png';self.other=self.folder/'B.png'
        cv2.imencode('.png',self.frame)[1].tofile(self.image)
        cv2.imencode('.png',self.frame+1)[1].tofile(self.other)
        with patch.object(gui.DesktopApp,'refresh_cameras',lambda self:None),patch.object(gui.DesktopApp,'refresh_serials',lambda self:None):
            self.root=gui.RootClass();self.root.tk.call('tk','scaling',2.0)
            self.root.withdraw();self.app=gui.DesktopApp(self.root)
        self.addCleanup(self.close_root)
        self.assertIsNotNone(getattr(self.app,'recognition',None),'Recognition-first session must exist')
        self.prediction={'task':'classification','top3':[{'id':9,'label':'corn','score':.7},{'id':1,'label':'bacon','score':.2},{'id':6,'label':'potato','score':.1}]}

    def close_root(self):
        for item in self.root.tk.call('after','info'):self.root.after_cancel(item)
        self.root.destroy()

    def pump(self,predicate):
        deadline=time.monotonic()+5
        while not predicate():
            self.root.update();time.sleep(.01)
            if time.monotonic()>deadline:self.fail('Timed out waiting for UI operation')
        self.root.update()

    def engine(self,predict=None):
        self.app.model=types.SimpleNamespace(path=Path('SYNTHETIC.onnx'),mode='oven_round2',labels=['a']*12,height=224,width=224,
            contract={'source_checkpoint_sha256':'c'*64},predict=predict or (lambda frame:self.prediction))
        self.app.loaded_identity={'display_name':'SYNTHETIC model','model_sha256':'a'*64,'classes_sha256':'b'*64,'checkpoint_sha256':'c'*64,
                                  'mode':'oven_round2','input_shape':[1,3,224,224]}
        return self.app.model

    def test_home_is_default_and_details_are_collapsed(self):
        self.assertEqual(self.app.tabs.tab(self.app.tabs.select(),'text'),'识别')
        self.assertFalse(self.app.presentation.logs_open.get())
        self.assertIsNone(self.app.model);self.assertIsNone(self.app.worker)

    def test_import_and_predict_without_camera(self):
        self.app.open_image_file(self.image);self.engine()
        self.assertFalse(self.app.camera_ready)
        self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        self.assertEqual(self.app.recognition.record['prediction'],self.prediction)
        self.assertEqual(self.app.recognition.record['source']['kind'],'file')
        self.assertEqual(self.app.presentation.result_name.get(),'corn')

    def test_switch_a_b_a_while_inference_runs_rejects_old_result_and_duplicate_click(self):
        self.app.open_image_file(self.image);release=threading.Event();calls=[]
        self.addCleanup(release.set)
        def slow(frame):
            calls.append(frame.copy());release.wait(3);return self.prediction
        self.engine(slow);self.app.infer_current();self.app.infer_current()
        self.app.open_image_file(self.other);self.app.open_image_file(self.image)
        release.set();self.pump(lambda:not self.app.inference_running)
        self.assertEqual(len(calls),1);self.assertIsNone(self.app.recognition.record)
        self.assertEqual(self.app.presentation.result_name.get(),'尚未识别')

    def test_config_change_invalidates_ready_and_previous_result(self):
        self.app.open_image_file(self.image);self.engine();self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        self.app.network_size.set('320')
        self.assertIsNone(self.app.model);self.assertIsNone(self.app.recognition.record)
        self.assertEqual(self.app.presentation.result_name.get(),'尚未识别')

    def test_corrupt_image_clears_success_and_shows_inline_error(self):
        self.app.open_image_file(self.image);self.engine();self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        bad=self.folder/'corrupt.png';bad.write_bytes(b'SYNTHETIC invalid image')
        self.app.open_image_file(bad)
        self.assertIsNone(self.app.recognition.record);self.assertIsNone(self.app.recognition.source_frame)
        self.assertIn('图片',self.app.presentation.feedback.get())

    def test_disconnected_camera_refused_but_does_not_clear_imported_image(self):
        self.app.open_image_file(self.image);self.engine();self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        record=self.app.recognition.record
        self.app.events.put(('camera_stopped',None));self.app._poll()
        self.assertIs(self.app.recognition.record,record)
        self.app.use_camera_source();self.app.infer_current()
        self.assertFalse(self.app.inference_running);self.assertIsNone(self.app.recognition.record)
        self.assertIn('画面',self.app.presentation.feedback.get())

    def test_model_configuration_changed_during_load_cannot_become_ready(self):
        model=self.folder/'model.onnx';model.write_bytes(b'SYNTHETIC model')
        labels=self.folder/'classes.json';labels.write_text('[]')
        self.app.model_file.set(str(model));self.app.label_file.set(str(labels))
        self.app.model_task.set(gui.OVEN_ROUND2_MODE_LABEL)
        release=threading.Event();self.addCleanup(release.set)
        engine=types.SimpleNamespace(path=model,labels=['x']*12,height=224,width=224,mode='oven_round2',contract={'source_checkpoint_sha256':'c'*64})
        def slow(*args):release.wait(3);return engine
        with patch.object(gui,'ModelEngine',side_effect=slow):
            self.app.load_model();self.app.model_file.set(str(self.folder/'changed.onnx'))
            release.set();self.pump(lambda:not self.app.model_loading)
        self.assertIsNone(self.app.model);self.assertIn('配置',self.app.presentation.feedback.get())

    def test_full_log_queue_does_not_drop_completion(self):
        self.app.open_image_file(self.image);self.engine()
        while not self.app.events.full():self.app.events.put(('log','SYNTHETIC noisy log'))
        self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        self.assertIsNotNone(self.app.recognition.record)

    def test_restore_same_configuration_during_load_still_rejects_old_generation(self):
        model=self.folder/'model.onnx';model.write_bytes(b'SYNTHETIC model')
        labels=self.folder/'classes.json';labels.write_text('[]')
        self.app.model_file.set(str(model));self.app.label_file.set(str(labels));self.app.model_task.set(gui.OVEN_ROUND2_MODE_LABEL)
        release=threading.Event();self.addCleanup(release.set)
        engine=types.SimpleNamespace(path=model,labels=['x']*12,height=224,width=224,mode='oven_round2',contract={'source_checkpoint_sha256':'c'*64})
        def slow(*args):release.wait(3);return engine
        with patch.object(gui,'ModelEngine',side_effect=slow):
            self.app.load_model();self.app.model_file.set('changed');self.app.model_file.set(str(model))
            release.set();self.pump(lambda:not self.app.model_loading)
        self.assertIsNone(self.app.model)

    def test_camera_queue_cannot_repopulate_disconnected_source(self):
        self.app.use_camera_source();self.app.camera_ready=True
        self.app.latest_frame=self.frame.copy();self.app.frames.put((self.frame.copy(),time.time()))
        self.app.events.put(('camera_stopped',None));self.app._poll()
        self.assertIsNone(self.app.latest_frame);self.assertIsNone(self.app._recognition_frame())

    def test_ended_camera_worker_clears_source_even_if_stop_event_was_lost(self):
        self.app.use_camera_source();self.app.camera_ready=True;self.app.latest_frame=self.frame.copy()
        self.app.worker=types.SimpleNamespace(is_alive=lambda:False)
        self.app._poll()
        self.assertFalse(self.app.camera_ready);self.assertIsNone(self.app._recognition_frame())

    def test_camera_frames_do_not_replace_imported_image(self):
        self.app.open_image_file(self.image);self.app.camera_ready=True
        self.app.frames.put((self.frame+1,time.time()));self.app._poll()
        self.assertTrue(np.array_equal(self.app._recognition_frame(),self.frame))

    def test_prediction_failure_clears_previous_success_and_reports_reason(self):
        self.app.open_image_file(self.image);self.engine();self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        def fail(frame):raise RuntimeError('SYNTHETIC inference failure')
        self.engine(fail);self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        self.assertIsNone(self.app.recognition.record)
        self.assertEqual(self.app.presentation.result_name.get(),'尚未识别')
        self.assertIn('SYNTHETIC inference failure',self.app.presentation.feedback.get())

    def test_primary_actions_and_inline_error_are_inside_small_window(self):
        self.root.deiconify();self.root.geometry('1366x768');self.root.update()
        for widget in [self.app.presentation.recognize_button,self.app.presentation.save_result_button,self.app.presentation.feedback_label]:
            with self.subTest(widget=str(widget)):
                self.assertTrue(widget.winfo_ismapped())
                self.assertLessEqual(widget.winfo_rooty()-self.root.winfo_rooty()+widget.winfo_height(),self.root.winfo_height())

    def test_result_time_source_and_model_are_visible_at_150_percent(self):
        self.app.open_image_file(self.image);self.engine();self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        self.root.deiconify();self.root.geometry('1366x768');self.root.update()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child;yield from descendants(child)
        for variable in [self.app.presentation.result_time,self.app.presentation.result_source,self.app.presentation.result_model]:
            widget=next(child for child in descendants(self.app.presentation.result_card)
                        if 'textvariable' in child.keys() and str(child.cget('textvariable'))==str(variable))
            with self.subTest(variable=str(variable)):
                self.assertTrue(widget.winfo_ismapped())
                self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),self.app.presentation.result_card.winfo_rooty()+self.app.presentation.result_card.winfo_height())

    def test_long_manual_labels_and_model_name_cannot_hide_result_identity(self):
        self.root.deiconify();self.root.geometry('1366x768');self.root.update()
        self.app.open_image_file(self.image);self.engine()
        self.app.loaded_identity['display_name']='手动模型'*40+'.onnx'
        self.prediction['top3'][0]['label']='这是一个很长的人工类别名称'*16
        self.app.infer_current();self.pump(lambda:not self.app.inference_running)
        ui=self.app.presentation
        self.assertGreater(ui.image_label.winfo_width(),300)
        for widget in [ui.source_label,ui.model_label,ui.result_thumbnail]:
            self.assertTrue(widget.winfo_ismapped(),str(widget))
            self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),ui.result_card.winfo_rooty()+ui.result_card.winfo_height())
        self.assertEqual(self.app.recognition.record['prediction']['top3'][0]['label'],self.prediction['top3'][0]['label'])

    def test_stale_success_cannot_replace_current_corrupt_image_error(self):
        self.app.open_image_file(self.image);release=threading.Event();self.addCleanup(release.set)
        def slow(frame):release.wait(3);return self.prediction
        self.engine(slow);self.app.infer_current()
        bad=self.folder/'corrupt.png';bad.write_bytes(b'SYNTHETIC invalid image')
        self.app.open_image_file(bad);feedback=self.app.presentation.feedback.get()
        release.set();self.pump(lambda:not self.app.inference_running)
        self.assertEqual(self.app.presentation.feedback.get(),feedback)
        self.assertIn('无法打开图片',feedback)


if __name__=='__main__':unittest.main()
