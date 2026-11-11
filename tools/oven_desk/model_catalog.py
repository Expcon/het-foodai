"""Optional local experiment list; choosing an entry never loads it automatically."""
import json,re
from pathlib import Path
from inference_engine import ModelContractError,load_oven_contract

MANUAL_LABEL='手动选择（保持当前配置）'


def _child(root,value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ModelContractError('Catalog paths must stay relative to the local bundle')
    path=(root/value).resolve()
    if not path.is_relative_to(root):raise ModelContractError('Catalog path escapes bundle')
    return path


def read_catalog(path):
    path=Path(path)
    if not path.exists():return {}
    try:obj=json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError) as exc:raise ModelContractError('Cannot read experiment catalog') from exc
    if not isinstance(obj,dict) or obj.get('format')!='het-oven-model-catalog/v1' or not isinstance(obj.get('models'),list):
        raise ModelContractError('Invalid experiment catalog format')
    root=path.resolve().parent;result={};ids=set()
    for entry in obj['models']:
        if not isinstance(entry,dict):raise ModelContractError('Invalid experiment entry')
        label=entry.get('label');ident=entry.get('id');sha=entry.get('checkpoint_sha256')
        if (not isinstance(label,str) or not label.strip() or label in result or label==MANUAL_LABEL
            or not isinstance(ident,str) or not ident or ident in ids
            or not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha)):
            raise ModelContractError('Duplicate/invalid experiment identity')
        if _child(root,entry.get('model')).suffix.lower()!='.onnx' or _child(root,entry.get('classes')).suffix.lower()!='.json':
            raise ModelContractError('Catalog supports approved twelve-class ONNX exports only')
        result[label]={**entry,'root':root};ids.add(ident)
    return result


def resolve_profile(entry):
    model=_child(entry['root'],entry['model']);classes=_child(entry['root'],entry['classes'])
    contract=load_oven_contract(classes,model,'oven_round2')
    if contract['source_checkpoint_sha256']!=entry['checkpoint_sha256']:
        raise ModelContractError('Experiment checkpoint differs from catalog identity')
    return dict(model=model,classes=classes,checkpoint_sha256=entry['checkpoint_sha256'])
