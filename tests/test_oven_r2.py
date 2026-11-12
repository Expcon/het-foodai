"""SYNTHETIC R2 manifest/batch-policy tests; no food accuracy evidence."""
import copy,importlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from torch.utils.data import DataLoader,SequentialSampler,RandomSampler,Dataset
from tools.oven_training.model import ROOT
from tools.oven_training.controlled import recipe_config,DEFAULT_RECIPE


class SyntheticClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.drop=torch.nn.Dropout(.25);self.linear=torch.nn.Linear(3,3)
    def forward(self,x):
        logits=self.linear(self.drop(self.bn(x).mean((2,3))))
        return logits if self.training else (logits.softmax(1),logits)


class R2Tests(unittest.TestCase):
    def batching(self):return importlib.import_module('tools.oven_training.batching')
    def r2(self):return importlib.import_module('tools.oven_training.r2')

    def test_1923_and_actual_1922_merge_tail_without_losing_or_repeating_samples(self):
        cls=self.batching().MergeShortTailBatchSampler
        for n,tail in [(1923,35),(1922,34),(67,35),(64,32),(72,8)]:
            sampler=cls(SequentialSampler(range(n)),32,8);batches=list(sampler)
            self.assertEqual(len(batches),len(sampler));self.assertEqual(batches[-1].__len__(),tail)
            self.assertEqual([i for batch in batches for i in batch],list(range(n)))
            self.assertGreaterEqual(min(map(len,batches)),8)

    def test_sampler_rejects_silent_drop_or_impossible_merge(self):
        cls=self.batching().MergeShortTailBatchSampler
        for args in [(3,32,8),(65,0,8),(65,32,0),(65,32,33)]:
            with self.assertRaises(ValueError):list(cls(SequentialSampler(range(args[0])),args[1],args[2]))

    def test_r2_recipe_is_explicit_and_does_not_unlock_original_controls(self):
        values={**DEFAULT_RECIPE,'lr':.0003,'tail_batch_policy':'merge_short_tail','tail_min_size':8}
        with self.assertRaises(ValueError):recipe_config(values)
        with self.assertRaises(ValueError):recipe_config(values,lr_control=True)
        self.assertEqual(recipe_config(values,r2_control=True),values)
        for update in [dict(lr=.001),dict(bn_policy='frozen_stats'),dict(batch_size=16),dict(drop_last=True),dict(tail_min_size=2)]:
            with self.assertRaises(ValueError):recipe_config({**values,**update},r2_control=True)
        self.assertEqual(recipe_config()['lr'],.001)

    def test_manifest_uses_explicit_visual_review_not_filename_and_preserves_other_classes(self):
        module=self.r2()
        def old(i,key,name):return dict(image_id=str(i),sha256=str(i)*64,pixel_sha256=chr(97+i)*64,
            class_key=key,class_id=9 if key=='corn' else 6,relative_path=name,new_split='train',source='new',group_id='old')
        original=[old(1,'corn','corn_one.png'),old(2,'corn','corn_half.png'),old(3,'potato','potato.png')]
        new=[dict(image_id='new',sha256='4'*64,pixel_sha256='e'*64,class_key='corn',class_id=9,relative_path='white.png',
                  status='ok',sidecar_sha_matches=True,sidecar_file_matches=True,sidecar_dimensions_match=True,group_id='new')]
        review={'1'*64:'half','2'*64:'whole','4'*64:'whole'}
        output=module.build_manifest(original,new,review)
        self.assertEqual(output[0]['new_split'],'excluded');self.assertEqual(output[1]['new_split'],'train')
        self.assertEqual(output[2],original[2]);self.assertEqual(output[3]['class_id'],9)
        self.assertEqual(original[0]['new_split'],'train')
        with self.assertRaises(ValueError):module.build_manifest(original,new,{})

    def test_unusable_or_duplicate_new_photos_never_enter_training(self):
        module=self.r2();old=[dict(image_id='old',sha256='a'*64,pixel_sha256='b'*64,class_key='potato',class_id=6,new_split='train')]
        row=dict(image_id='new',sha256='c'*64,pixel_sha256='d'*64,class_key='corn',class_id=9,relative_path='new.png',status='ok',
                 sidecar_sha_matches=True,sidecar_file_matches=True,sidecar_dimensions_match=True,group_id='new')
        result=module.build_manifest(old,[row],{'c'*64:'unusable_underexposed'})
        self.assertEqual(result[-1]['new_split'],'excluded')
        for change in [dict(pixel_sha256='b'*64),dict(sidecar_sha_matches=False)]:
            result=module.build_manifest(old,[{**row,**change}],{'c'*64:'whole'})
            self.assertEqual(result[-1]['new_split'],'excluded')

    def test_wrong_initialization_is_stopped_before_any_training_forward(self):
        from tools.oven_training.train import fit_exploratory
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-r2-init-',dir=ROOT/'runs') as directory:
            root=Path(directory);image=root/'image.png';Image.new('RGB',(32,24)).save(image)
            init=root/'initial';init.write_bytes(b'SYNTHETIC')
            recipe={**DEFAULT_RECIPE,'epochs':1,'lr':.0003,'tail_batch_policy':'merge_short_tail','tail_min_size':8}
            with patch('tools.oven_training.train.new_model',side_effect=lambda *a:SyntheticClassifier()),patch('tools.oven_training.train.configure_sdk'):
                with self.assertRaisesRegex(ValueError,'initialization'):
                    fit_exploratory([(image,['a','b','c'][i%3]) for i in range(35)],['a','b','c'],root/'run',init,'cpu','a'*64,
                        recipe=recipe,r2_control=True,observation={'comparison':'record_only'},
                        provenance={'reference_initialization_state_sha256':'0'*64})

    def test_record_only_fit_saves_complete_states_and_deterministic_tail_recovery(self):
        from tools.oven_training.train import fit_exploratory
        from tools.oven_training.observation import load_snapshot,resume_batches,restore_training
        from tools.oven_training.monitor import tensor_sha256,model_state_sha256
        cls=self.batching().MergeShortTailBatchSampler
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-r2-',dir=ROOT/'runs') as directory:
            root=Path(directory);image=root/'SYNTHETIC.png';Image.new('RGB',(80,50),(140,80,40)).save(image)
            init=root/'SYNTHETIC-init';init.write_bytes(b'SYNTHETIC')
            recipe={**DEFAULT_RECIPE,'epochs':1,'lr':.0003,'tail_batch_policy':'merge_short_tail','tail_min_size':8}
            with patch('tools.oven_training.train.new_model',side_effect=lambda *a:SyntheticClassifier()),patch('tools.oven_training.train.configure_sdk'):
                fit_exploratory([(image,['a','b','c'][i%3]) for i in range(67)],['a','b','c'],root/'run',init,'cpu','a'*64,
                    recipe=recipe,r2_control=True,observation={'comparison':'record_only','tail_epochs':(1,)})
            data=json.loads((root/'run/monitor/epoch_001.json').read_text())
            self.assertEqual(data['batch_trace']['sizes'],[32,35]);self.assertEqual(data['train_eval']['samples'],67)
            index=[json.loads(line) for line in (root/'run/observations/checkpoints.jsonl').read_text().splitlines()]
            self.assertEqual(len(index),4)
            before=next(r for r in index if r['file'].endswith('tail_pre.pt'));after=next(r for r in index if r['file'].endswith('tail_post.pt'))
            pre=load_snapshot(root/'run/observations'/before['file'],before['sha256'])
            post=load_snapshot(root/'run/observations'/after['file'],after['sha256'])
            self.assertEqual(len(pre['pending'][1]),35)
            model=SyntheticClassifier();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0005)
            # Recreate the same loader sampler; cached pending tensors avoid pretending to rebuild source augmentation.
            class Dummy(Dataset):
                def __len__(self):return 67
                def __getitem__(self,i):return torch.rand(3,224,224),i%3,i
            gen=torch.Generator().manual_seed(42)
            loader=DataLoader(Dummy(),batch_sampler=cls(RandomSampler(Dummy(),generator=gen),32,8),generator=gen)
            fixed=DataLoader(Dummy(),generator=torch.Generator().manual_seed(10042))
            restore_training(pre,model,optimizer,{'train':loader,'fixed':fixed},pre['binding'])
            images,labels,indices=next(resume_batches(loader,pre));self.assertTrue(torch.equal(images,pre['pending'][0]))
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(images),labels);loss.backward();optimizer.step()
            self.assertEqual(model_state_sha256(model),post['model_state_sha256'])
            self.assertEqual(pre['optimizer_state']['param_groups'][0]['lr'],.0003)


if __name__=='__main__':unittest.main()
