"""Fixed eval and BN evidence for exploratory training, never deployment."""
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import transform


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def model_state_sha256(model):
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        checksum.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def set_training_mode(model, policy):
    if policy not in {'update', 'frozen_stats'}:
        raise ValueError('Unknown BN running-statistics policy')
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats or module.running_mean is None:
                raise ValueError('Controlled experiment requires tracked BN statistics')
            module.train(policy == 'update')
            # Affine parameters keep identical optimizer membership in both arms.


def bn_snapshot(model):
    checksum = hashlib.sha256();layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        if module.running_mean is None or module.running_var is None:
            raise ValueError('BN has no saved running statistics')
        mean = module.running_mean.detach().cpu();var = module.running_var.detach().cpu()
        finite = bool(torch.isfinite(mean).all() and torch.isfinite(var).all())
        if not finite or bool((var < 0).any()):
            raise ValueError('Nonfinite BN mean/variance or negative variance')
        counter = int(module.num_batches_tracked)
        checksum.update(name.encode());checksum.update(mean.numpy().tobytes());checksum.update(var.numpy().tobytes())
        checksum.update(str(counter).encode())
        layers[name] = dict(running_mean=mean.tolist(), running_var=var.tolist(), num_batches_tracked=counter,
                            eps=module.eps, momentum=module.momentum, training=module.training,
                            track_running_stats=module.track_running_stats, finite=finite,
                            affine_trainable=bool(module.affine and module.weight.requires_grad and module.bias.requires_grad))
    return {'running_sha256': checksum.hexdigest(), 'layers': layers}


def bn_delta(before, after):
    if before['layers'].keys() != after['layers'].keys():
        raise ValueError('BN layers changed')
    layers = {}
    for name, old in before['layers'].items():
        new = after['layers'][name]
        mean = np.asarray(new['running_mean'])-np.asarray(old['running_mean'])
        var = np.asarray(new['running_var'])-np.asarray(old['running_var'])
        layers[name] = dict(counter_delta=new['num_batches_tracked']-old['num_batches_tracked'],
                            mean_l2=float(np.linalg.norm(mean)), var_l2=float(np.linalg.norm(var)),
                            mean_max_abs=float(np.abs(mean).max()), var_max_abs=float(np.abs(var).max()))
    return {'running_unchanged': before['running_sha256'] == after['running_sha256'], 'layers': layers}


class IndexedImages(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


class FixedImages(Dataset):
    """Reuse deterministic framed PIL cache; random augmentation is never cached."""
    def __init__(self, items): self.items = items;self.preprocess = transform(False)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        image, label = self.items[index]
        return self.preprocess(image), label


def evaluate_fixed(model, loader, keys, device, *, per_image=None):
    """No augmentation, eval mode, sample-weighted CE; restore RNG and mixed modes."""
    modes = [(module, module.training) for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    before = bn_snapshot(model);classes = len(keys)
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    class_loss = torch.zeros(classes, dtype=torch.float64)
    sample_index = 0
    try:
        model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                if labels.dtype != torch.int64 or bool((labels < 0).any()) or bool((labels >= classes).any()):
                    raise ValueError('Invalid fixed-eval target index')
                labels = labels.to(device)
                output = model(images.to(device))
                logits = output[1] if isinstance(output, tuple) else output
                if tuple(logits.shape) != (len(labels), classes) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Invalid fixed-eval logits')
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                if not bool(torch.isfinite(loss).all()): raise ValueError('Nonfinite fixed-eval loss')
                target = labels.cpu();predicted = logits.argmax(1).cpu()
                if per_image is not None:
                    raw = logits.detach().cpu();scores = raw.softmax(1)
                    order = raw.argsort(dim=1, descending=True, stable=True)
                    losses = loss.cpu()
                    for i in range(len(target)):
                        per_image.append(dict(sample_index=sample_index, target_id=int(target[i]),
                            input_sha256=tensor_sha256(images[i]), logits=raw[i].tolist(), scores=scores[i].tolist(),
                            top1_id=int(predicted[i]), top3_ids=order[i,:min(3,classes)].tolist(), loss=float(losses[i])))
                        sample_index += 1
                matrix += torch.bincount(target*classes+predicted, minlength=classes*classes).reshape(classes, classes)
                class_loss.scatter_add_(0, target, loss.double().cpu())
        after = bn_snapshot(model)
        if before['running_sha256'] != after['running_sha256']:
            raise ValueError('Fixed eval changed BN running buffers')
        counts = matrix.sum(1);samples = int(counts.sum());correct = int(matrix.diag().sum())
        if not samples: raise ValueError('Fixed eval requires nonempty data')
        return dict(samples=samples, correct=correct, loss=float(class_loss.sum())/samples, top1=correct/samples,
                    class_order=list(keys), confusion_matrix=matrix.tolist(), bn_unchanged=True,
                    scope='candidate-label diagnostic; not an independent generalization accuracy',
                    per_class={key:dict(samples=int(counts[i]),correct=int(matrix[i,i]),
                        loss=float(class_loss[i])/int(counts[i]) if counts[i] else None,
                        top1=int(matrix[i,i])/int(counts[i]) if counts[i] else None) for i,key in enumerate(keys)})
    finally:
        for module, training in modes: module.training = training
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
