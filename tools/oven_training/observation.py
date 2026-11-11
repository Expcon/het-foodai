"""Opt-in R0 evidence and safe recovery; not a production checkpoint format."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .model import class_digest, digest, output_dir
from .monitor import bn_snapshot, model_state_sha256


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if value is None or type(value) in (str, int, float, bool): return value
    raise ValueError('Unsafe checkpoint value type: ' + type(value).__name__)


def capture_rng():
    kind, keys, position, gaussian, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=kind, keys=keys.tolist(), position=position,
                gaussian=gaussian, cached=float(cached)), torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def restore_rng(state):
    n = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['gaussian'], n['cached']))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs from checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def state_digest(state):
    checksum = hashlib.sha256()
    for name, value in state.items():
        checksum.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def atomic_new(path, writer):
    """Hard-link publication is atomic and never replaces an existing destination."""
    path = output_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path.name)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('xb') as stream:
        writer(stream); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_json_new(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    atomic_new(path, lambda stream: stream.write(data))


class LoaderCursor:
    """Observe the stock loader; never replace or pre-consume its sampler."""
    def __init__(self, loader):
        if loader.num_workers != 0 or loader.generator is None:
            raise ValueError('Exact R0 recovery requires workers=0 and an explicit loader generator')
        self.start = loader.generator.get_state().clone()
        self.batches = 0
        self.indices = []

    def record(self, indices):
        self.batches += 1
        self.indices.extend(int(i) for i in indices)

    def state(self, loader):
        return dict(epoch_start_generator=self.start.clone(), batches_fetched=self.batches,
                    fetched_indices=list(self.indices), generator_current=loader.generator.get_state().clone())


def save_snapshot(path, model, optimizer, *, epoch, global_step, phase, binding, loaders, cursor, pending=None):
    if phase not in {'epoch_end', 'pre_forward', 'post_step'}:
        raise ValueError('Unknown observation boundary')
    if (phase == 'pre_forward') != (pending is not None):
        raise ValueError('Only pre-forward snapshots carry a fetched, unprocessed batch')
    if any(loader.num_workers != 0 or loader.generator is None for loader in loaders.values()):
        raise ValueError('Recovery requires explicit worker-free loader generators')
    state = cpu_tree(model.state_dict())
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('Nonfinite persistent model state')
    payload = dict(format='oven-r0-observation/v1', model_state=state, model_state_sha256=state_digest(state),
        optimizer_state=cpu_tree(optimizer.state_dict()), epoch=epoch, global_step=global_step, phase=phase,
        binding=cpu_tree(binding), rng=capture_rng(), loader_generators={k:v.generator.get_state().clone() for k,v in loaders.items()},
        cursor=cpu_tree(cursor), pending=cpu_tree(pending),
        module_modes={name:module.training for name,module in model.named_modules()},
        bn_attributes={name:{k:v[k] for k in ['eps','momentum','track_running_stats']} for name,v in bn_snapshot(model)['layers'].items()})
    atomic_new(path, lambda stream: torch.save(payload, stream))
    sha = digest(path)
    atomic_new(Path(str(path)+'.sha256'), lambda stream: stream.write((sha+'\n').encode('ascii')))
    return sha


def load_snapshot(path, expected_sha):
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or digest(path) != expected_sha:
        raise ValueError('Observation checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != 'oven-r0-observation/v1':
        raise ValueError('Not a safe R0 observation checkpoint')
    if state_digest(payload['model_state']) != payload['model_state_sha256']:
        raise ValueError('Persistent model fingerprint mismatch')
    return payload


def restore_training(payload, model, optimizer, loaders, expected_binding):
    if payload['binding'] != expected_binding or set(loaders) != set(payload['loader_generators']):
        raise ValueError('Recovery binding or loader set mismatch')
    attributes = {name:{k:v[k] for k in ['eps','momentum','track_running_stats']} for name,v in bn_snapshot(model)['layers'].items()}
    if attributes != payload['bn_attributes']:
        raise ValueError('Recovery BN attributes differ')
    model.load_state_dict(payload['model_state'], strict=True)
    # Optimizer tensors can alias an in-memory CPU payload without this copy.
    optimizer.load_state_dict(copy.deepcopy(payload['optimizer_state']))
    for name, module in model.named_modules(): module.training = payload['module_modes'][name]
    for key, loader in loaders.items(): loader.generator.set_state(payload['loader_generators'][key])
    restore_rng(payload['rng'])


def resume_batches(loader, payload):
    """Rebuild the consumed stock iterator, then restore RNG before new augmentation."""
    cursor = payload['cursor']
    if cursor is None:
        restore_rng(payload['rng'])
        yield from loader
        return
    loader.generator.set_state(cursor['epoch_start_generator'])
    iterator = iter(loader); seen = []
    for _ in range(cursor['batches_fetched']):
        _, _, indices = next(iterator); seen.extend(int(i) for i in indices)
    if seen != cursor['fetched_indices'] or not torch.equal(loader.generator.get_state(), cursor['generator_current']):
        raise ValueError('Stock DataLoader replay indices or generator differ')
    restore_rng(payload['rng'])
    if payload['pending'] is not None: yield payload['pending']
    yield from iterator


def mix_states(model, weights, statistics):
    """W includes BN affine. S includes only tracked BN buffers, never affine."""
    expected = model.state_dict()
    params = set(dict(model.named_parameters()))
    buffers = {name + ('.' if name else '') + field for name,module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for field in ['running_mean','running_var','num_batches_tracked']}
    if set(expected) != params | buffers or set(weights) != set(expected) or set(statistics) != set(expected):
        raise ValueError('Unknown/missing persistent keys in W/S experiment')
    mixed = {}
    for name, value in expected.items():
        for state in [weights, statistics]:
            if state[name].shape != value.shape or state[name].dtype != value.dtype:
                raise ValueError('W/S shape or dtype mismatch: '+name)
        mixed[name] = (weights if name in params else statistics)[name].detach().clone()
    return mixed


def parameter_copies(model):
    return {name: p.detach().cpu().clone() for name,p in model.named_parameters()}


def gradient_evidence(model):
    values = {'head':0., 'backbone':0.}
    for name,p in model.named_parameters():
        if p.grad is None: continue
        gradient = p.grad.detach().cpu().double()
        if not bool(torch.isfinite(gradient).all()): raise ValueError('Nonfinite gradient')
        values['head' if name.startswith('model.10.') else 'backbone'] += float(gradient.square().sum())
    return {key: value**.5 for key,value in values.items()}


class ObservationRun:
    def __init__(self, out, model, optimizer, loaders, config, entries, *, reference=None, tail_epochs=(24,27,28,29,30)):
        self.out = output_dir(out); self.out.mkdir(parents=True, exist_ok=False)
        self.reference = Path(reference) if reference is not None else None
        self.tail_epochs = set(tail_epochs); self.global_step = 0
        self.model = model; self.optimizer = optimizer; self.loaders = loaders
        self.source = [dict(sample_index=i, source_sha256=digest(path), path=str(path), class_key=key) for i,(path,key) in enumerate(entries)]
        from . import train, monitor
        self.binding = dict(config=config, class_list_sha256=class_digest(config['class_keys']),
            sources=self.source, torch_version=str(torch.__version__), numpy_version=np.__version__,
            cuda_version=torch.version.cuda, device=str(next(model.parameters()).device),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,
            source_sha256={Path(p).name:digest(p) for p in [__file__,train.__file__,monitor.__file__]})
        self.records = {}
        if self.reference is not None:
            if config != json.loads((self.reference/'fixed_config.json').read_text(encoding='utf-8')):
                raise ValueError('R0 recipe/config differs from original A')
            for e in range(config['epochs']+1):
                self.records[e]=json.loads((self.reference/'monitor'/f'epoch_{e:03d}.json').read_text(encoding='utf-8'))
            initial=json.loads((self.reference/'monitor/initialization.json').read_text(encoding='utf-8'))
            if model_state_sha256(model)!=initial['model_state_sha256']:
                raise ValueError('R0 initialization differs from original A')
            with (self.reference/'history.csv').open(encoding='utf-8',newline='') as stream:
                import csv
                self.history={int(r['epoch']):r for r in csv.DictReader(stream)}
        write_json_new(self.out/'binding.json', self.binding)
        self.index=[]

    def save(self, name, epoch, phase, cursor=None, pending=None):
        path=self.out/'checkpoints'/name
        sha=save_snapshot(path,self.model,self.optimizer,epoch=epoch,global_step=self.global_step,phase=phase,
            binding=self.binding,loaders=self.loaders,cursor=cursor,pending=pending)
        # Read back every checkpoint with the restricted loader immediately.
        loaded=load_snapshot(path,sha)
        if loaded['model_state_sha256']!=model_state_sha256(self.model): raise ValueError('Saved state differs')
        record=dict(file=path.relative_to(self.out).as_posix(),sha256=sha,epoch=epoch,phase=phase,global_step=self.global_step,
                    bytes=path.stat().st_size,model_state_sha256=loaded['model_state_sha256'])
        with (self.out/'checkpoints.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
        self.index.append(record)

    def batch(self, epoch, batch_index, trace):
        if self.reference is not None and trace!=self.records[epoch]['batch_trace']['batches'][batch_index]:
            raise ValueError(f'R0 batch/input mismatch at epoch {epoch} batch {batch_index}')

    def step(self, epoch, batch_index, before, gradient, loss, correct):
        changes={'head':0.,'backbone':0.}
        for name,p in self.model.named_parameters():
            delta=p.detach().cpu().double()-before[name].double()
            if not bool(torch.isfinite(delta).all()): raise ValueError('Nonfinite parameter update')
            changes['head' if name.startswith('model.10.') else 'backbone']+=float(delta.square().sum())
        row=dict(epoch=epoch,batch_index=batch_index,global_step=self.global_step,loss=loss,correct=correct,
            gradient_l2=gradient,parameter_update_l2={k:v**.5 for k,v in changes.items()},
            optimizer_lr=[group['lr'] for group in self.optimizer.param_groups])
        with (self.out/'batch_diagnostics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def evaluation(self, epoch, metrics, evidence):
        if len(evidence)!=len(self.source): raise ValueError('Incomplete per-image fixed eval')
        if self.reference is not None and metrics!=self.records[epoch]['train_eval']:
            raise ValueError(f'R0 fixed-eval mismatch at epoch {epoch}')
        for row,source in zip(evidence,self.source):
            if row['sample_index']!=source['sample_index'] or metrics['class_order'][row['target_id']]!=source['class_key']:
                raise ValueError('Fixed eval source/label order differs')
            row.update(source)
        data=''.join(json.dumps(row,allow_nan=False)+'\n' for row in evidence).encode('utf-8')
        atomic_new(self.out/'eval'/f'epoch_{epoch:03d}.jsonl',lambda stream:stream.write(data))
        write_json_new(self.out/'eval'/f'epoch_{epoch:03d}_metrics.json',metrics)
        self.save(f'epoch_{epoch:03d}.pt',epoch,'epoch_end')

    def online(self, row):
        if self.reference is None:return
        old=self.history[row['epoch']]
        for key in ['train_loss','train_accuracy_augmented','train_eval_loss','train_eval_top1']:
            if row[key]!=float(old[key]): raise ValueError(f'R0 online metric {key} differs at epoch {row["epoch"]}')

    def complete(self):
        if self.reference is not None:
            old=torch.load(self.reference/'last.pt',map_location='cpu',weights_only=True)['model_state']
            now=self.model.state_dict()
            if list(old)!=list(now) or any(not torch.equal(old[k],now[k].detach().cpu()) for k in old):
                raise ValueError('R0 final persistent tensors differ from A')
        for record in self.index:
            if digest(self.out/record['file'])!=record['sha256']:raise ValueError('Saved checkpoint changed')
        write_json_new(self.out/'gate.json',dict(status='PASS',scope='seen-training-image exact reproduction; not independent accuracy',
            reference_bound=self.reference is not None,epochs=self.binding['config']['epochs'],global_steps=self.global_step,
            checkpoints=len(self.index),final_persistent_tensors=len(self.model.state_dict()),
            final_model_state_sha256=model_state_sha256(self.model),all_reference_metrics_and_batches_equal=self.reference is not None))
