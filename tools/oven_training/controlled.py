"""Prepare BN controls by default; fitting requires explicit --run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from .model import PREPROCESS, PRETRAINED_SHA, digest, output_dir
from .round2 import V2_KEYS, resolve_image
from .export_desk import write_json

DEFAULT_RECIPE = dict(format='oven-bn-control/v1', epochs=30, batch_size=32, seed=42,
                      optimizer='AdamW', lr=.001, weight_decay=.0005, amp=False, workers=0,
                      drop_last=False, bn_policy='update', bn_affine_trainable=True,
                      preprocessing=PREPROCESS,
                      augmentation='horizontal_flip_p0.5;color_jitter_brightness_contrast_saturation_0.1')


def read_rows(path):
    if path.suffix.lower() != '.csv':
        rows=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Rows must be a list of records')
        return rows
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for field in ['physical_identity_verified','session_identity_verified','trusted_label']:
            value=row.get(field,'').strip().lower()
            if value not in {'true','false',''}:raise ValueError('Explicit true/false required for '+field)
            row[field]=value=='true'
        row['class_id']=int(row['class_id'])
    return rows


def recipe_config(values=None, *, lr_control=False, r2_control=False):
    values = {} if values is None else values
    extra={'tail_batch_policy','tail_min_size'} if r2_control else set()
    if lr_control and r2_control:raise ValueError('R1 single-variable and R2 data/tail experiments are distinct')
    if not isinstance(values, dict) or set(values)-set(DEFAULT_RECIPE)-extra: raise ValueError('Unknown recipe fields')
    result = {**DEFAULT_RECIPE, **values}
    if r2_control:
        if result.get('tail_batch_policy')!='merge_short_tail' or type(result.get('tail_min_size')) is not int or result['tail_min_size']!=8:
            raise ValueError('R2 requires explicit merge_short_tail / min8')
    if type(result['epochs']) is not int or result['epochs'] < 1: raise ValueError('Epochs must be a positive integer')
    if result['bn_policy'] not in {'update','frozen_stats'}: raise ValueError('Unknown BN statistics policy')
    if lr_control or r2_control:
        if type(result['lr']) is not float or result['lr'] != .0003 or result['bn_policy'] != 'update':
            raise ValueError('Registered R1 requires lr=0.0003 and normal BN updates')
    for key, value in DEFAULT_RECIPE.items():
        if key == 'lr' and (lr_control or r2_control): continue
        if key not in {'epochs','bn_policy'} and (type(result[key]) is not type(value) or result[key] != value):
            raise ValueError('Controlled recipe fixes '+key)
    return result


def validation_readiness(train, val, keys):
    """Conservative provenance gate; a declaration is not proof of unseen real objects."""
    problems = Counter()
    if not train: problems['empty_training_pool'] += 1
    if not val: problems['no_validation_candidates'] += 1
    for split, rows in [('train',train),('val',val)]:
        for row in rows:
            if row.get('new_split') != split: problems[split+'_wrong_split'] += 1
            key = row.get('class_key')
            if key not in keys or type(row.get('class_id')) is not int or row['class_id'] != keys.index(key):
                problems[split+'_class_mapping_invalid'] += 1
            for field in ['sha256','pixel_sha256']:
                if not re.fullmatch('[0-9a-f]{64}',str(row.get(field,''))): problems[split+'_'+field+'_missing'] += 1
            for field in ['group_id','physical_sample_id','session_id','identity_reviewed_by','identity_reviewed_at']:
                if not isinstance(row.get(field),str) or row[field].strip().lower() in {'','unknown','unassigned'}:
                    problems[split+'_'+field+'_missing'] += 1
            for field in ['physical_identity_verified','session_identity_verified']:
                if row.get(field) is not True: problems[split+'_'+field+'_unverified'] += 1
            if split == 'val':
                if row.get('trusted_label') is not True: problems['val_label_unconfirmed'] += 1
                for field in ['label_reviewed_by','label_reviewed_at']:
                    if not isinstance(row.get(field),str) or row[field].strip().lower() in {'','unknown'}:
                        problems['val_'+field+'_missing'] += 1
    for field in ['sha256','pixel_sha256','group_id','physical_sample_id','session_id']:
        left={str(row.get(field)).strip() for row in train if row.get(field)}
        right={str(row.get(field)).strip() for row in val if row.get(field)}
        overlap=left & right
        if overlap: problems[field+'_cross_split'] += len(overlap)
    for field in ['sha256','pixel_sha256']:
        occurrences=Counter(row.get(field) for row in val if row.get(field))
        duplicates={value:n for value,n in occurrences.items() if n>1}
        if not duplicates:continue
        problems['val_'+field+'_duplicate_records']+=sum(n-1 for n in duplicates.values())
        labels={value:{row.get('class_key') for row in val if row.get(field)==value} for value in duplicates}
        conflicts=sum(len(names)>1 for names in labels.values())
        if conflicts:problems['val_'+field+'_conflicting_labels']+=conflicts
    counts=Counter(row.get('class_key') for row in val)
    for key in keys:
        if not counts[key]:problems['validation_class_missing'] += 1
    return dict(status='BLOCKED' if problems else 'READY_FOR_EXPLORATORY_MONITORING',
                issues=dict(problems), train_samples=len(train), val_samples=len(val),
                class_order=list(keys), val_per_class={k:counts[k] for k in keys},
                independent_validation_accuracy=None, formal_generalization_claim_allowed=False,
                limitation='Human identity/label declarations and duplicate-group audit must be verified; readiness alone does not freeze classes or certify generalization.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True);p.add_argument('--validation-rows',type=Path)
    p.add_argument('--run',action='store_true');p.add_argument('--device',default='cuda:0')
    observation_mode=p.add_mutually_exclusive_group()
    observation_mode.add_argument('--observe-r0',action='store_true',help='Record full states and require exact original-A replay')
    observation_mode.add_argument('--observe-r1',action='store_true',help='Registered lr-only control against completed exact R0')
    p.add_argument('--reference-run',type=Path,help='Original A training directory; read only')
    args=p.parse_args();recipe=recipe_config(json.loads(args.recipe.read_text(encoding='utf-8')),lr_control=args.observe_r1)
    if (args.observe_r0 or args.observe_r1) != (args.reference_run is not None):p.error('Observation mode and --reference-run must be supplied together')
    all_rows=read_rows(args.rows);rows=[r for r in all_rows if r.get('new_split')!='excluded']
    seen=set();entries=[]
    for row in rows:
        key=row.get('class_key')
        if row.get('new_split')!='train' or key not in V2_KEYS or row.get('class_id')!=V2_KEYS.index(key) or not row.get('group_id'):
            raise ValueError('Only explicit grouped twelve-class train rows allowed')
        if not re.fullmatch('[0-9a-f]{64}',str(row.get('pixel_sha256',''))) or row['pixel_sha256'] in seen:
            raise ValueError('Missing pixel hash or unremoved training pixel duplicate')
        path=resolve_image(row,args.source_root)
        if digest(path)!=row['sha256']:raise ValueError('Training original changed')
        seen.add(row['pixel_sha256']);entries.append((path,key))
    if set(key for _,key in entries)!=set(V2_KEYS):raise ValueError('Every experimental class requires training data')
    val=read_rows(args.validation_rows) if args.validation_rows else []
    readiness=validation_readiness(rows,val,V2_KEYS)
    if args.validation_rows and readiness['status']=='BLOCKED':raise ValueError('Supplied validation is not independent/qualified: '+str(readiness['issues']))
    val_entries=[]
    for row in val:
        path=resolve_image(row,args.source_root)
        if digest(path)!=row['sha256']:raise ValueError('Validation original changed')
        val_entries.append((path,row['class_key']))
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Pinned official pretrained SHA mismatch')
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'prepared.json',dict(state='PREPARED_EXPLORATORY_ONLY',recipe=recipe,class_keys=V2_KEYS,
        train_rows_sha256=digest(args.rows),validation_rows_sha256=digest(args.validation_rows) if args.validation_rows else None,
        train_samples=len(rows),per_class=dict(Counter(key for _,key in entries)),validation=readiness,
        pretrained_sha256=PRETRAINED_SHA,training_requested=args.run))
    print(json.dumps(dict(state='PREPARED_EXPLORATORY_ONLY',train_samples=len(rows),validation_status=readiness['status'],training_requested=args.run)))
    if args.run:
        from .train import fit_exploratory
        fit_exploratory(entries,V2_KEYS,out/'training',args.pretrained,args.device,digest(args.rows),
                        provenance={'scope':'controlled exploratory BN statistics experiment; not formal accuracy'},
                        recipe=recipe,validation_entries=val_entries,validation_report=readiness,
                        observation={'reference':args.reference_run,'comparison':'lr_only' if args.observe_r1 else 'exact'}
                                    if args.observe_r0 or args.observe_r1 else None)
    return 0


if __name__=='__main__':raise SystemExit(main())
