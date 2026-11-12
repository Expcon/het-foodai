"""Whole-cob R2: explicit new manifest, fixed recipe, no independent-val fallback."""
import argparse,copy,json
from collections import Counter
from pathlib import Path
from .controlled import DEFAULT_RECIPE,recipe_config,read_rows,validation_readiness
from .model import PRETRAINED_SHA,digest,output_dir
from .round2 import V2_KEYS,resolve_image
from .round2_audit import safe_child
from .observation import write_json_new

R2_RECIPE={**DEFAULT_RECIPE,'lr':.0003,'tail_batch_policy':'merge_short_tail','tail_min_size':8}


def build_manifest(old,new,forms):
    """Forms must be separately visually reviewed and SHA-bound, never filename guesses."""
    result=copy.deepcopy(old);pixels=set();hashes=set()
    for row in result:
        if row.get('new_split')!='train':continue
        if row.get('class_key')=='corn':
            form=forms.get(row['sha256'])
            if form not in {'whole','half'}:raise ValueError('Old corn needs explicit complete shape review')
            row['r2_visual_form']=form
            if form=='half':row.update(new_split='excluded',r2_exclusion_reason='confirmed_half_outside_whole_cob_scope')
        if row['new_split']=='train':pixels.add(row['pixel_sha256']);hashes.add(row['sha256'])
    for original in new:
        row=copy.deepcopy(original);form=forms.get(row['sha256'])
        if form not in {'whole','half','unusable_underexposed'}:raise ValueError('New image requires actual shape/quality review')
        if row.get('class_key')!='corn' or row.get('class_id')!=9:raise ValueError('White corn must retain ID9')
        reasons=[]
        if form!='whole':reasons.append(form)
        if row.get('status')!='ok' or not all(row.get(k) is True for k in ['sidecar_sha_matches','sidecar_file_matches','sidecar_dimensions_match']):reasons.append('decode_or_sidecar_integrity')
        if row.get('pixel_sha256') in pixels or row['sha256'] in hashes:reasons.append('duplicate_training_input')
        row.update(new_split='excluded' if reasons else 'train',r2_exclusion_reason=';'.join(reasons),r2_visual_form=form,
                   source='r2_whole_corn',previously_used_for_development=True,unused_final_test_eligible=False)
        if not reasons:pixels.add(row['pixel_sha256']);hashes.add(row['sha256'])
        result.append(row)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ['rows','old-source-root','new-source-root','pretrained','out','reference-initialization']:parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--expect-rows-sha256',required=True);parser.add_argument('--run',action='store_true');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if digest(args.rows)!=args.expect_rows_sha256:raise ValueError('Approved R2 manifest SHA changed')
    if digest(args.pretrained)!=PRETRAINED_SHA:raise ValueError('Official initialization SHA changed')
    reference_initial=json.loads(args.reference_initialization.read_text('utf-8'))
    expected_initial=reference_initial['model_state_sha256']
    if reference_initial['pretrained_sha256']!=PRETRAINED_SHA:raise ValueError('Initialization reference is not the same official pretrained specimen')
    rows=read_rows(args.rows);active=[r for r in rows if r.get('new_split')=='train'];seen=set();entries=[]
    for row in active:
        key=row.get('class_key')
        if key not in V2_KEYS or row.get('class_id')!=V2_KEYS.index(key) or not row.get('group_id'):raise ValueError('Explicit fixed12 train mapping/group required')
        if not row.get('pixel_sha256') or row['pixel_sha256'] in seen:raise ValueError('Missing or duplicate training pixels')
        if key=='corn' and row.get('r2_visual_form')!='whole':raise ValueError('R2 corn training requires actual whole-cob review')
        path=safe_child(args.new_source_root,row['relative_path']) if row.get('source')=='r2_whole_corn' else resolve_image(row,args.old_source_root)
        if digest(path)!=row['sha256']:raise ValueError('Source changed after audit')
        seen.add(row['pixel_sha256']);entries.append((path,key))
    if set(key for _,key in entries)!=set(V2_KEYS):raise ValueError('All12 training classes required')
    recipe=recipe_config(R2_RECIPE,r2_control=True);validation=validation_readiness(active,[],V2_KEYS)
    from .batching import MergeShortTailBatchSampler
    from torch.utils.data import SequentialSampler
    sizes=[len(batch) for batch in MergeShortTailBatchSampler(SequentialSampler(range(len(active))),32,8)]
    out=output_dir(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json_new(out/'prepared.json',dict(status='EXPLORATORY_R2_ONLY',training_requested=args.run,recipe=recipe,
        rows_sha256=digest(args.rows),samples=len(active),per_class=dict(Counter(key for _,key in entries)),class_keys=V2_KEYS,
        batches_per_epoch=len(sizes),batch_sizes=sizes,original_unmerged_tail=len(active)%32,no_dropped_or_duplicated_samples=True,
        validation=validation,comparison_scope='data/scope/tail-policy differ from R1; NOT a single-variable control'))
    print(json.dumps(dict(prepared=True,samples=len(active),batches=len(sizes),tail=sizes[-1],run=args.run)),flush=True)
    if args.run:
        from .train import fit_exploratory
        fit_exploratory(entries,V2_KEYS,out/'training',args.pretrained,args.device,digest(args.rows),recipe=recipe,
            r2_control=True,validation_report=validation,observation={'comparison':'record_only'},
            provenance={'scope':'whole-cob R2 exploratory fit; no independent validation; exposed new cohort becomes training data',
                'reference_initialization_state_sha256':expected_initial,'reference_initialization_file_sha256':digest(args.reference_initialization)})
    return 0


if __name__=='__main__':raise SystemExit(main())
