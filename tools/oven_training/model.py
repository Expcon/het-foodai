"""Experimental YOLO contract; separate from the frozen MVP predictor API."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PREPROCESS = dict(size=224, color='RGB', resize='bilinear-contain-center-pad', pad=114, scale='1/255', layout='NCHW')
PRETRAINED_SHA = 'c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def class_digest(keys):
    if not isinstance(keys, list) or not keys or any(not isinstance(k,str) or not re.fullmatch(r'[a-z][a-z0-9_]*',k) for k in keys) or len(set(keys))!=len(keys):
        raise ValueError('Class keys must be unique explicit lowercase labels')
    return hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()


def validate_metadata(meta):
    if not isinstance(meta,dict) or meta.get('format')!='oven-yolo-exploratory/v1' or meta.get('architecture')!='yolo11n-cls':
        raise ValueError('Not a supported experimental checkpoint')
    if meta.get('class_list_sha256')!=class_digest(meta.get('class_keys')):
        raise ValueError('Ordered classes/hash mismatch')
    if meta.get('preprocess')!=PREPROCESS or meta.get('independent_validation') is not False:
        raise ValueError('Preprocessing/scope mismatch')
    if meta.get('ultralytics_version')!='8.3.230' or type(meta.get('epoch')) is not int or meta['epoch']<1:
        raise ValueError('Unverified SDK/epoch')


def input_path(relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative or '..' in Path(relative).parts:
        raise ValueError('Invalid candidate image path')
    path=(ROOT/relative).resolve()
    if not path.is_relative_to((ROOT/'data/raw').resolve()):
        raise ValueError('Candidate must be in ignored data/raw')
    return path


def check_rows(rows,keys):
    class_digest(keys)
    active=[]
    for row in rows:
        if row.get('new_split')=='excluded':
            continue
        if row.get('new_split')!='train' or row.get('class_key') not in keys or not row.get('group_id'):
            raise ValueError('Exploratory fit accepts only grouped train rows with explicit labels')
        input_path(row.get('relative_path'))
        if not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')):
            raise ValueError('Image SHA256 required')
        active.append(row)
    if not active:
        raise ValueError('No usable training images')
    return active


def output_dir(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Outputs must stay in ignored runs/')
    return path


def configure_sdk(local=None):
    local=output_dir(local) if local is not None else ROOT/'runs/oven-first-round-20261106/sdk-settings'
    local.mkdir(parents=True,exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(local),MPLCONFIGDIR=str(local/'matplotlib'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from tools.edge_export.offline_guard import install
    install()


def frame_image(image):
    from PIL import Image,ImageOps
    rgb=ImageOps.exif_transpose(image).convert('RGB')
    resized=ImageOps.contain(rgb,(224,224),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(224,224),(114,114,114))
    canvas.paste(resized,((224-resized.width)//2,(224-resized.height)//2))
    return canvas


def transform(train=False):
    from torchvision import transforms as T
    parts=[frame_image]
    if train:
        parts.extend([T.RandomHorizontalFlip(.5),T.ColorJitter(brightness=.1,contrast=.1,saturation=.1)])
    return T.Compose([*parts,T.ToTensor()])


def new_model(keys,pretrained=None):
    import ultralytics
    from ultralytics.nn.tasks import ClassificationModel
    if ultralytics.__version__!='8.3.230':
        raise ValueError('Use the pinned Ultralytics 8.3.230 SDK')
    ultralytics.settings.update({'sync':False})
    if pretrained:
        if digest(pretrained)!=PRETRAINED_SHA:
            raise ValueError('Official pretrained specimen SHA mismatch')
        # Only this exact previously verified upstream file uses the SDK object loader.
        model=ultralytics.YOLO(str(pretrained),task='classify').model.float()
        ClassificationModel.reshape_outputs(model,len(keys))
    else:
        model=ClassificationModel('yolo11n-cls.yaml',nc=len(keys),verbose=False)
    model.names=dict(enumerate(keys))
    for parameter in model.parameters():parameter.requires_grad_(True)
    return model


def load_model(path,device='cpu',expected_sha=None):
    import torch
    if expected_sha and digest(path)!=expected_sha:
        raise ValueError('Checkpoint SHA mismatch')
    try:
        payload=torch.load(path,map_location='cpu',weights_only=True)
    except (OSError,EOFError,RuntimeError,pickle.UnpicklingError) as exc:
        raise ValueError('model not ready: checkpoint is unreadable by the safe weights loader') from exc
    if not isinstance(payload,dict) or not isinstance(payload.get('metadata'),dict) or not isinstance(payload.get('model_state'),dict):
        raise ValueError('model not ready: checkpoint payload must contain metadata and model_state dictionaries')
    meta=payload['metadata']
    validate_metadata(meta)
    model=new_model(meta['class_keys'])
    model.load_state_dict(payload['model_state'],strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        probs=model(torch.zeros(1,3,224,224,device=device))[0]
    if tuple(probs.shape)!=(1,len(meta['class_keys'])) or not torch.isfinite(probs).all():
        raise ValueError('Model forward verification failed')
    return model,meta
