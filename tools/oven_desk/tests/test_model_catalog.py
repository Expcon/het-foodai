"""SYNTHETIC catalog boundary/state tests; never real model acceptance."""
import hashlib,json,queue,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import oven_capture_gui as gui
from inference_engine import ModelContractError


class Value:
    def __init__(self,value=''):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'experimental_models.json'
        self.model=self.root/'models/R1/model.onnx';self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b'SYNTHETIC catalog fixture; not an ONNX model')
        self.classes=self.model.parent/'classes.json';self.classes.write_text('{}')
        self.entry=dict(id='R1',label='R1 experimental',model='models/R1/model.onnx',classes='models/R1/classes.json',
                        checkpoint_sha256='a'*64)
        self.write([self.entry])

    def write(self,models):self.path.write_text(json.dumps(dict(format='het-oven-model-catalog/v1',models=models)),encoding='utf-8')

    def module(self):
        import model_catalog
        return model_catalog

    def test_missing_catalog_preserves_manual_startup(self):
        module=self.module()
        self.assertEqual(module.read_catalog(self.root/'missing.json'),{})
        self.assertEqual(module.MANUAL_LABEL,'手动选择（保持当前配置）')

    def test_catalog_rejects_duplicate_labels_bad_schema_and_path_escape(self):
        module=self.module()
        for entries in [[self.entry,self.entry], [{**self.entry,'model':'../escape.onnx'}],
                        [{**self.entry,'model':str(self.model.resolve())}], [{**self.entry,'checkpoint_sha256':'bad'}]]:
            self.write(entries)
            with self.assertRaises(ModelContractError):module.read_catalog(self.path)

    def test_resolve_uses_existing_safe_contract_and_exact_checkpoint_identity(self):
        module=self.module();entry=module.read_catalog(self.path)['R1 experimental']
        with patch.object(module,'load_oven_contract',return_value={'source_checkpoint_sha256':'a'*64}) as verify:
            result=module.resolve_profile(entry)
        verify.assert_called_once_with(self.classes.resolve(),self.model.resolve(),'oven_round2')
        self.assertEqual(result['model'],self.model.resolve())
        with patch.object(module,'load_oven_contract',return_value={'source_checkpoint_sha256':'b'*64}):
            with self.assertRaises(ModelContractError):module.resolve_profile(entry)

    def strict_entry(self):
        return {**self.entry,'onnx_sha256':hashlib.sha256(self.model.read_bytes()).hexdigest(),
            'classes_sha256':hashlib.sha256(self.classes.read_bytes()).hexdigest(),
            'mode':'oven_round2','input_shape':[1,3,224,224],'output_shape':[1,12],'dtype':'float32'}

    def test_strict_catalog_pins_both_files_and_input_spec_without_breaking_legacy(self):
        module=self.module();self.write([self.strict_entry()]);entry=module.read_catalog(self.path)['R1 experimental']
        with patch.object(module,'load_oven_contract',return_value={'source_checkpoint_sha256':'a'*64}):
            module.resolve_profile(entry)
            self.classes.write_text('{"SYNTHETIC":"changed"}')
            with self.assertRaises(ModelContractError):module.resolve_profile(entry)
        self.write([self.entry]);self.assertIn('R1 experimental',module.read_catalog(self.path))

    def test_partial_or_wrong_shape_catalog_pins_are_rejected(self):
        module=self.module()
        for entry in [{**self.entry,'onnx_sha256':'a'*64},{**self.strict_entry(),'input_shape':[1,3,320,320]},
                      {**self.strict_entry(),'dtype':'float16'},{**self.strict_entry(),'classes_sha256':'bad'}]:
            self.write([entry])
            with self.assertRaises(ModelContractError):module.read_catalog(self.path)

    def app(self):
        app=gui.DesktopApp.__new__(gui.DesktopApp)
        app.model_profile=Value('R1 experimental');app.last_model_profile=self.module().MANUAL_LABEL
        app.model_profiles=self.module().read_catalog(self.path)
        app.model_file=Value('old.onnx');app.label_file=Value('old.json');app.model_task=Value('old')
        app.network_size=Value('320');app.model_state=Value('ready');app.model=object()
        app.model_loading=False;app.inference_running=False;app._log=lambda text:None
        return app

    def test_busy_selection_cannot_replace_loaded_model_or_fields(self):
        app=self.app();old=app.model;app.inference_running=True
        app.select_model_profile()
        self.assertIs(app.model,old);self.assertEqual(app.model_file.get(),'old.onnx')
        self.assertEqual(app.model_profile.get(),self.module().MANUAL_LABEL)

    def test_select_is_explicit_clears_ready_and_does_not_open_runtime(self):
        app=self.app()
        with patch.object(gui,'resolve_profile',return_value=dict(model=self.model,classes=self.classes,checkpoint_sha256='a'*64)):
            app.select_model_profile()
        self.assertIsNone(app.model);self.assertEqual(app.network_size.get(),'224')
        self.assertEqual(app._model_mode(),'oven_round2');self.assertIn('未就绪',app.model_state.get())
        self.assertFalse(app.model_loading)

    def test_profile_mismatch_is_rejected_instead_of_loading_other_weight(self):
        app=self.app();app.events=queue.Queue()
        with patch.object(gui,'resolve_profile',return_value=dict(model=self.model,classes=self.classes,checkpoint_sha256='a'*64)), \
             patch.object(gui.messagebox,'showwarning') as warning,patch.object(gui,'ModelEngine') as engine:
            app.load_model()
        engine.assert_not_called();warning.assert_called_once();self.assertIsNone(app.model)

    def test_worker_rechecks_checkpoint_identity_before_ready(self):
        app=self.app();app.events=queue.Queue()
        app.model_file.set(str(self.model));app.label_file.set(str(self.classes))
        app.model_task.set(gui.OVEN_ROUND2_MODE_LABEL);app.network_size.set('224')
        with patch.object(gui,'resolve_profile',return_value=dict(model=self.model,classes=self.classes,checkpoint_sha256='a'*64)), \
             patch.object(gui,'ModelEngine',return_value=types.SimpleNamespace(contract={'source_checkpoint_sha256':'b'*64})), \
             patch.object(gui.threading,'Thread',side_effect=lambda target,**kw:types.SimpleNamespace(start=target)):
            app.load_model()
        kind,_=app.events.get_nowait();self.assertEqual(kind,'model_error')


if __name__=='__main__':unittest.main()
