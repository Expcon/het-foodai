"""Frozen checkpoint mode diagnosis. No optimizer, gradients, or state updates."""
import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import random

from .error_audit import BASE, CHECKPOINT_SHA
from .export_desk import write_json
from .model import ROOT, configure_sdk, digest, frame_image, load_model, output_dir, transform
from .round2 import V2_KEYS, resolve_image


def state_digest(model):
    """Cover parameters AND persistent buffers, including BN counters."""
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        checksum.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def configure_mode(model, mode):
    import torch
    model.train(mode in {'train_batch_bn', 'train_running_bn'})
    use_batch = mode in {'train_batch_bn', 'eval_batch_bn'}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train(use_batch)
            if use_batch:
                # Same training normalization, with no running/counter writes.
                module.track_running_stats = False
    return model


def probabilities(output):
    """Pinned Classify semantics: train logits; eval (softmax, logits)."""
    import torch
    if isinstance(output, tuple):
        probability, logits = output
        if not torch.allclose(probability, logits.softmax(1), atol=1e-6, rtol=1e-5):
            raise ValueError('Eval probability/logit semantics differ')
    else:
        logits = output
        probability = logits.softmax(1)
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [batch, classes] logits')
    return probability, logits


def original_dataset(entries, keys):
    """Execute ONLY the real nested Dataset class, never the trainer function."""
    from PIL import Image
    from torch.utils.data import Dataset
    source = ROOT/'tools/oven_training/train.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    trainer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_exploratory')
    node = next(node for node in trainer.body if isinstance(node, ast.ClassDef) and node.name == 'Images')
    scope = dict(Dataset=Dataset, Image=Image, frame_image=frame_image, transform=transform, entries=entries, keys=keys)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), scope)
    return scope['Images'](), ast.unparse(node)


def epoch_batches(samples, epoch=30, batch_size=32):
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(range(samples), batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    for _ in range(epoch):
        batches = [indices.tolist() for indices in loader]
    return batches


def metrics(predictions, labels, rows):
    groups = {}
    for key in V2_KEYS:
        ids = [i for i, row in enumerate(rows) if row['class_key'] == key]
        groups[key] = {'n': len(ids), 'directory_agreement': sum(predictions[i] == labels[i] for i in ids)}
    for source in ['round1', 'new']:
        for key in ['potato', 'sausage']:
            ids = [i for i, row in enumerate(rows) if row['class_key'] == key and row['source'] == source]
            groups[f'{source}_{key}'] = {'n': len(ids), 'directory_agreement': sum(predictions[i] == labels[i] for i in ids)}
    return {'n': len(labels), 'directory_agreement': sum(a == b for a, b in zip(predictions, labels)), 'groups': groups}


def forward_pass(model, inputs, batches, device, capture_bn=False):
    import numpy as np
    import torch
    result = torch.empty((len(inputs), len(V2_KEYS)), dtype=torch.float32)
    observed = {}
    hooks = []
    if capture_bn:
        def hook(name):
            def record(module, args):
                x = args[0].detach()
                count = x.numel() // x.shape[1]
                mean = x.mean((0, 2, 3)).double().cpu().numpy()
                var = x.var((0, 2, 3), unbiased=False).double().cpu().numpy()
                item = observed.setdefault(name, {'n': 0, 'sum': np.zeros_like(mean), 'sumsq': np.zeros_like(mean)})
                item['n'] += count
                item['sum'] += count * mean
                item['sumsq'] += count * (var + mean*mean)
            return record
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d): hooks.append(module.register_forward_pre_hook(hook(name)))
    before = state_digest(model)
    with torch.inference_mode():
        for batch_index, indices in enumerate(batches):
            p, _ = probabilities(model(inputs[indices].to(device)))
            result[indices] = p.cpu()
            if (batch_index+1) % 20 == 0: print(f'Forward-only batches {batch_index+1}/{len(batches)}', flush=True)
    for handle in hooks: handle.remove()
    after = state_digest(model)
    if before != after: raise ValueError('Model parameters or buffers changed during forward')
    bn_report = []
    modules = dict(model.named_modules())
    for name, item in observed.items():
        module = modules[name]
        mean = item['sum'] / item['n']
        var = np.maximum(0, item['sumsq']/item['n'] - mean*mean)
        stored_mean = module.running_mean.detach().cpu().numpy()
        stored_var = module.running_var.detach().cpu().numpy()
        bn_report.append({'name': name, 'normalized_mean_rmse': float(np.sqrt(np.mean((mean-stored_mean)**2/(stored_var+module.eps)))),
            'observed_to_stored_var_median': float(np.median(var/(stored_var+module.eps))),
            'stored_var_min': float(stored_var.min()), 'stored_var_max': float(stored_var.max()),
            'eps': module.eps, 'momentum': module.momentum, 'num_batches_tracked': int(module.num_batches_tracked),
            'finite': bool(np.isfinite(stored_mean).all() and np.isfinite(stored_var).all()),
            'negative_var_channels': int((stored_var < 0).sum())})
    return result, {'before': before, 'after': after, 'unchanged': True}, bn_report


def run(args):
    out = output_dir(args.out)
    out.mkdir(parents=True, exist_ok=False)
    configure_sdk(out/'sdk-settings')
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rows_path = BASE/'audit/combined_rows.json'
    all_rows = json.loads(rows_path.read_text(encoding='utf-8'))
    rows = [r for r in all_rows if r['new_split'] == 'train']
    entries = [(resolve_image(row, args.source_root), row['class_key']) for row in rows]
    protected_paths = [BASE/'yolo11n-12cls-fixed30/last.pt', BASE/'onnx/oven12.onnx', BASE/'onnx/classes.json',
                       ROOT/'runs/oven-first-round-20261106/yolo11n-fixed30/last.pt',
                       ROOT/'tools/oven_training/train.py', ROOT/'tools/oven_training/model.py', rows_path]
    before_files = {str(path.relative_to(ROOT)): digest(path) for path in protected_paths}
    write_json(out/'protected_before.json', before_files)
    for index, ((path, _), row) in enumerate(zip(entries, rows)):
        if digest(path) != row['sha256']: raise ValueError('Original training photo changed')
    dataset, dataset_source = original_dataset(entries, V2_KEYS)
    (out/'actual_dataset_source.py.txt').write_text(dataset_source, encoding='utf-8')
    augmented_description = repr(dataset.augment)
    dataset.augment = transform(False)
    tensors, labels = [], []
    for index in range(len(dataset)):
        image, label = dataset[index]
        if label != rows[index]['class_id'] or V2_KEYS[label] != rows[index]['class_key']:
            raise ValueError('Actual Dataset target differs from training row')
        tensors.append(image); labels.append(label)
    inputs = torch.stack(tensors)
    del tensors
    batches = epoch_batches(len(rows))
    # Actual collation, including integer target dtype, without random augmentation.
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0)
    for indices, (images, target) in zip(batches, loader):
        if target.dtype != torch.int64 or target.tolist() != [labels[i] for i in indices] or not torch.equal(images, inputs[indices]):
            raise ValueError('Actual DataLoader input or target differs')
    write_json(out/'loader_audit.json', {'status': 'PASS', 'samples': len(rows), 'class_counts': dict(Counter(row['class_key'] for row in rows)),
        'targets_dtype': 'torch.int64', 'criterion': 'CrossEntropyLoss on raw logits', 'potato_target': V2_KEYS.index('potato'),
        'sausage_target': V2_KEYS.index('sausage'), 'batches': len(batches), 'last_batch_size': len(batches[-1]),
        'epoch30_order_reconstructed': True, 'last_batch_labels': dict(Counter(V2_KEYS[labels[i]] for i in batches[-1])),
        'same_inputs_verified': True, 'augmentation_disabled_for_comparison': True, 'actual_augmentation': augmented_description})
    outputs = {}; summary = {}; preservation = {}; bn_reports = {}
    for mode in ['eval', 'train_batch_bn', 'train_running_bn', 'eval_batch_bn']:
        print('Independent frozen copy: '+mode, flush=True)
        model, metadata = load_model(BASE/'yolo11n-12cls-fixed30/last.pt', args.device, CHECKPOINT_SHA)
        if digest(rows_path) != metadata['config']['rows_sha256']: raise ValueError('Training version differs')
        configure_mode(model, mode)
        p, preserved, bn_report = forward_pass(model, inputs, batches, args.device, capture_bn=True)
        outputs[mode] = p
        summary[mode] = metrics(p.argmax(1).tolist(), labels, rows)
        preservation[mode] = preserved
        bn_reports[mode] = bn_report
        if mode == 'eval':
            (out/'installed_head_forward.txt').write_text(inspect.getsource(type(model.model[-1]).forward), encoding='utf-8')
            write_json(out/'dropout.json', [{'name': n, 'p': m.p} for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout)])
        del model
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    comparisons = {}
    for a,b in [('eval','train_running_bn'),('train_batch_bn','eval_batch_bn'),('eval','train_batch_bn')]:
        comparisons[f'{a}__{b}'] = {'max_abs_probability_difference': float((outputs[a]-outputs[b]).abs().max()),
                                    'top1_agree': int((outputs[a].argmax(1)==outputs[b].argmax(1)).sum())}
    records = []
    for i,row in enumerate(rows):
        record = {key: row.get(key) for key in ['image_id','relative_path','class_key','class_id','source','sha256','foil','shelf','physical_sample_id']}
        record['modes'] = {}
        for mode,p in outputs.items():
            scores, ids = p[i].topk(3)
            record['modes'][mode] = {'probabilities': p[i].tolist(), 'top3': [{'id': n,'key': V2_KEYS[n],'score': v} for n,v in zip(ids.tolist(),scores.tolist())]}
        records.append(record)
    write_json(out/'per_image_modes.json', records)
    write_json(out/'bn_observed_stats.json', bn_reports)
    write_json(out/'model_state_preservation.json', preservation)
    after_files = {str(path.relative_to(ROOT)): digest(path) for path in protected_paths}
    if after_files != before_files: raise ValueError('Protected file changed')
    write_json(out/'summary.json', {'status': 'PASS', 'device': args.device, 'modes': summary, 'comparisons': comparisons,
        'protected_files_unchanged': True, 'parameters_and_buffers_unchanged': True,
        'scope': 'Frozen-state, unaugmented training-image diagnosis; candidate labels; no independent accuracy'})
    print(json.dumps({'modes': {mode: {'all': (value['directory_agreement'],value['n']),
            'old_potato': value['groups']['round1_potato'], 'new_potato': value['groups']['new_potato'],
            'sausage': value['groups']['sausage']} for mode,value in summary.items()}, 'comparisons': comparisons}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    return run(parser.parse_args())


if __name__ == '__main__': raise SystemExit(main())
