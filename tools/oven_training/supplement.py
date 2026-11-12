"""Read-only qualification then frozen-model comparison. No training/export action."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.audit_images import BKTree, IMAGE_EXTENSIONS, image_facts, write_csv
from scripts.propose_groups import UnionFind
from .model import PREPROCESS, class_digest, configure_sdk, digest, load_model, output_dir, transform
from .observation import write_json_new
from .round2 import V2_KEYS
from .round2_audit import safe_child, scene_evidence

FOLDERS={'三文鱼':'salmon','土豆':'potato','香肠':'sausage','玉米_白':'corn'}
V2_SHA='44878014afe8381055ed2a26fffff6269982e3f5b57c2da0fbedc596127770d5'
R1_SHA='fec9b89f29e65dc5f35976e704ebce9900a9c34a9b1c231f5adfdd93b02813cd'
SEALED={'test','external_test','final_test','sealed_test','frozen_test'}


def known(value):
    return isinstance(value,str) and value.strip().lower() not in {'','unknown','none','null','tbd','unassigned'}


def qualify(rows, training):
    """Evidence declarations are prerequisites, never inferred from dates or IDs."""
    result=copy.deepcopy(rows);union=UnionFind(r['image_id'] for r in result);seen={}
    for row in result:
        for field in ['sha256','pixel_sha256','declared_group','physical_sample_id','session_id','parent_source_group_id']:
            value=row.get(field)
            if known(value):
                key=(field,value)
                if key in seen:union.join(row['image_id'],seen[key])
                else:seen[key]=row['image_id']
        for peer in row.get('near_peer_ids',[]):
            if peer in union.parent:union.join(row['image_id'],peer)
    groups=defaultdict(list)
    for row in result:groups[union.find(row['image_id'])].append(row)
    training_complete=bool(training) and all(t.get('physical_identity_verified') is True and t.get('session_identity_verified') is True
        and all(known(t.get(f)) for f in ['physical_sample_id','session_id','parent_source_group_id']) for t in training)
    training_ids={f:{t[f] for t in training if known(t.get(f))} for f in ['physical_sample_id','session_id','parent_source_group_id']}
    for members in groups.values():
        sealed=any(str(r.get('intended_use','')).lower() in SEALED or r.get('reserved_final_test') is True for r in members)
        overlap=any(r.get('training_byte_matches') or r.get('training_pixel_matches') or
                    any(known(r.get(f)) and r[f] in training_ids[f] for f in training_ids) for r in members)
        group='review_'+hashlib.sha256('\n'.join(sorted(r['image_id'] for r in members)).encode()).hexdigest()[:20]
        for row in members:
            reasons=[]
            if not training_complete:reasons.append('training_identity_incomplete')
            if overlap:reasons.append('training_overlap_in_review_component')
            if row.get('decode_status')!='OK':reasons.append('unreadable_image')
            if row.get('candidate_class_key') not in V2_KEYS:reasons.append('unsupported_candidate_class')
            if row.get('physical_identity_verified') is not True:reasons.append('physical_identity_unverified')
            if row.get('session_identity_verified') is not True:reasons.append('session_identity_unverified')
            for field in ['physical_sample_id','session_id','parent_source_group_id','source_evidence','identity_reviewed_by','identity_reviewed_at',
                          'label_reviewed_by','label_reviewed_at','label_evidence']:
                if not known(row.get(field)):reasons.append(field+'_missing')
            if row.get('trusted_label') is not True:reasons.append('human_label_unconfirmed')
            if row.get('training_independence_verified') is not True or not known(row.get('training_independence_evidence')):
                reasons.append('training_independence_unverified')
            if row.get('label_conflict'):reasons.append('human_label_directory_conflict')
            if row.get('near_training_matches',0) and row.get('near_relation_reviewed') is not True:reasons.append('near_training_relation_unresolved')
            if row.get('intended_use') not in {'dev','val','development_validation'}:reasons.append('development_role_not_declared')
            row.update(review_component=group,qualification_reasons=reasons,
                route='sealed_excluded' if sealed else 'qualified_dev' if not reasons else 'diagnostic_only',
                independent_physical_count=1 if not reasons and not sealed else None)
    return result


def _scene(sidecar, filename):
    scene=scene_evidence(sidecar,filename);context=sidecar.get('camera_context',{})
    foil=scene['foil']['value'];shelf=scene['shelf']['value']
    background=context.get('background',sidecar.get('background','unknown'))
    # No foil does not establish a black tray, and horizontal center is not middle shelf.
    if background=='unknown' and foil=='present':background='foil_candidate_from_filename'
    return dict(foil=foil,shelf=shelf,background=background,lighting=scene['lighting']['value'],
                physical_sample_id=scene['physical_sample_id'],session_id=scene['session_id'],oven_id=scene['oven_id'],
                scene_evidence='source metadata/explicit filename tokens only; not visually or physically verified')


def audit(source, training, out, *, prior_seen=None, heldout_sha=(), registrations=()):
    source=Path(source).resolve(strict=True);out=output_dir(out)
    if out.is_relative_to(source):raise ValueError('Audit output overlaps source')
    out.mkdir(parents=True,exist_ok=False)
    before={p.relative_to(source).as_posix():digest(safe_child(source,p.relative_to(source).as_posix())) for p in sorted(source.rglob('*')) if p.is_file()}
    write_json_new(out/'source_snapshot.json',before)
    train_bytes=defaultdict(list);train_pixels=defaultdict(list);tree=BKTree()
    for i,r in enumerate(training):
        name=r.get('image_id',str(i));train_bytes[r['sha256']].append(name);train_pixels[r['pixel_sha256']].append(name)
        if r.get('phash'):tree.add(int(r['phash'],16),name)
    registered={r['sha256']:r for r in registrations}
    if len(registered)!=len(registrations):raise ValueError('Duplicate registration SHA rows require review')
    rows=[];near_train=[];near_peer=[];peer_tree=BKTree();prior_seen=prior_seen or {}
    for relative in before:
        if Path(relative).suffix.lower() not in IMAGE_EXTENSIONS:continue
        path=safe_child(source,relative);parts=Path(relative).parts;key=FOLDERS.get(parts[0],'unknown')
        image_id='supp_'+hashlib.sha256(relative.encode()).hexdigest()[:24]
        sidecar_path=path.with_suffix('.json');metadata={};metadata_error=None
        if sidecar_path.is_file():
            try:metadata=json.loads(sidecar_path.read_text(encoding='utf-8-sig'))
            except (OSError,ValueError) as exc:metadata_error=type(exc).__name__
        if not isinstance(metadata,dict):metadata={};metadata_error='metadata_not_object'
        context=metadata.get('camera_context',{})
        if not isinstance(context,dict):context={}
        row=dict(image_id=image_id,relative_path=relative,sha256=before[relative],candidate_class_key=key,
            class_id=V2_KEYS.index(key) if key in V2_KEYS else None,
            declared_group='/'.join(parts[:2]) if len(parts)>2 else parts[0],
            decode_status='OK',pixel_sha256=None,phash=None,width=None,height=None,format=None,
            metadata_present=sidecar_path.is_file(),metadata_error=metadata_error,
            captured_at=metadata.get('created_host_local','unknown'),filesystem_mtime_ns=path.stat().st_mtime_ns,
            physical_identity_verified=False,session_identity_verified=False,trusted_label=False,
            parent_source_group_id='unknown',source_evidence='',identity_reviewed_by='',identity_reviewed_at='',
            label_reviewed_by='',label_reviewed_at='',label_evidence='',intended_use='diagnostic',
            training_independence_verified=False,training_independence_evidence='',
            source_label_status=metadata.get('label_status','missing'),source_human_label_confirmed=context.get('human_label_confirmed'),
            near_relation_reviewed=False,near_peer_ids=[],**_scene(metadata,path.name))
        try:
            with Image.open(path) as image:
                image.load();row.update(image_facts(ImageOps.exif_transpose(image)));row['format']=image.format
                row['image_metadata_keys']=sorted(image.info);row['exif_tag_ids']=sorted(image.getexif().keys())
        except (OSError,ValueError,Image.DecompressionBombError) as exc:
            row.update(decode_status='FAIL',decode_error=type(exc).__name__)
        record=registered.get(row['sha256'])
        if record:
            for field in ['physical_sample_id','session_id','parent_source_group_id','source_evidence','identity_reviewed_by','identity_reviewed_at',
                          'label_reviewed_by','label_reviewed_at','label_evidence','intended_use','reserved_final_test',
                          'physical_identity_verified','session_identity_verified','trusted_label','near_relation_reviewed','background','foil','shelf',
                          'training_independence_verified','training_independence_evidence']:
                if field in record:row[field]=record[field]
            row['label_conflict']=record.get('confirmed_class_key',key)!=key
        if row['sha256'] in set(heldout_sha):row['intended_use']='final_test'
        row['training_byte_matches']=train_bytes[row['sha256']];row['training_pixel_matches']=train_pixels[row['pixel_sha256']]
        row['previous_diagnostic_evidence']=sorted(set(prior_seen.get(row['sha256'],[])+prior_seen.get('pixel:'+str(row['pixel_sha256']),[])))
        matches=list(tree.find(int(row['phash'],16),6)) if row['phash'] else []
        row['near_training_matches']=len(matches)
        near_train.extend(dict(candidate=image_id,training=old,distance=distance,relation='unreviewed_perceptual_candidate') for old,distance in matches)
        if row['phash']:
            for previous,distance in peer_tree.find(int(row['phash'],16),6):
                near_peer.append(dict(left=previous,right=image_id,distance=distance,relation='review_candidate_not_identity_proof'))
                row['near_peer_ids'].append(previous)
            peer_tree.add(int(row['phash'],16),image_id)
        rows.append(row)
    rows=qualify(rows,training)
    write_json_new(out/'inventory.json',rows);write_json_new(out/'near_training.json',near_train);write_json_new(out/'near_peer.json',near_peer)
    groups=Counter(r['declared_group'] for r in rows)
    plan=dict(status='QUALIFIED_DEV_AVAILABLE' if any(r['route']=='qualified_dev' for r in rows) else 'BLOCKED',
        source_root=str(source),inventory_sha256=digest(out/'inventory.json'),source_snapshot_sha256=digest(out/'source_snapshot.json'),
        roles=dict(Counter(r['route'] for r in rows)),images=len(rows),declared_folder_groups=dict(groups),
        independent_physical_objects=None,policy='No split inferred from dates/folder numbering; sealed components excluded before prediction. Unknown identities remain diagnostics.',
        final_test_assignments=[r['image_id'] for r in rows if r['route']=='sealed_excluded'],
        qualified_dev=[r['image_id'] for r in rows if r['route']=='qualified_dev'])
    write_json_new(out/'pre_prediction_plan.json',plan)
    (out/'pre_prediction_plan.sha256').write_text(digest(out/'pre_prediction_plan.json')+'\n',encoding='ascii')
    summary=dict(images=len(rows),classes=dict(Counter(r['candidate_class_key'] for r in rows)),formats=dict(Counter(str(r['format']) for r in rows)),
        resolutions=dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),decode_failures=sum(r['decode_status']!='OK' for r in rows),
        metadata_files=sum(r['metadata_present'] for r in rows),training_byte_overlap=sum(bool(r['training_byte_matches']) for r in rows),
        training_pixel_overlap=sum(bool(r['training_pixel_matches']) for r in rows),previously_diagnosed=sum(bool(r['previous_diagnostic_evidence']) for r in rows),
        near_training_pairs=len(near_train),near_peer_pairs=len(near_peer),near_threshold=6,
        exact_duplicate_groups=sum(n>1 for n in Counter(r['sha256'] for r in rows).values()),
        pixel_duplicate_groups=sum(n>1 for n in Counter(r['pixel_sha256'] for r in rows if r['pixel_sha256']).values()),
        qualification_reasons=dict(Counter(reason for r in rows for reason in r['qualification_reasons'])),plan=plan)
    write_json_new(out/'audit_summary.json',summary)
    missing=[dict(declared_folder_group=group,images=count,candidate_class_key=next(r['candidate_class_key'] for r in rows if r['declared_group']==group),
             physical_sample_id='',parent_source_group_id='',source_batch_id='',source_evidence='',session_id='',
             identity_reviewed_by='',identity_reviewed_at='',human_confirmed_label='',label_reviewed_by='',label_reviewed_at='',
             background='',shelf='',intended_use='',notes='Numbered folder is not verified identity; fill from actual records only') for group,count in sorted(groups.items())]
    write_csv(out/'missing_evidence_intake.csv',missing)
    after={p.relative_to(source).as_posix():digest(safe_child(source,p.relative_to(source).as_posix())) for p in sorted(source.rglob('*')) if p.is_file()}
    if after!=before:raise ValueError('Source changed during audit')
    return rows


def score(records):
    def metrics(rows,model,qualified):
        matrix=[[0]*12 for _ in range(12)];per_class={key:dict(images=0,top1_matches=0,top3_contains=0,
            physical_objects=0 if qualified else None,independent_source_groups=0 if qualified else None) for key in V2_KEYS}
        for row in rows:
            key=row['candidate_class_key']
            if key not in V2_KEYS:continue
            target=V2_KEYS.index(key);prediction=row[model];matrix[target][prediction['top1_id']]+=1
            counts=per_class[key];counts['images']+=1;counts['top1_matches']+=int(target==prediction['top1_id']);counts['top3_contains']+=int(target in prediction['top3_ids'])
        labelled=sum(r['images'] for r in per_class.values());correct=sum(r['top1_matches'] for r in per_class.values());top3=sum(r['top3_contains'] for r in per_class.values())
        result=dict(images=len(rows),candidate_label_images=labelled,per_class=per_class,confusion=matrix,
            top1=correct/labelled if qualified and labelled else None,top3=top3/labelled if qualified and labelled else None,
            physical_objects=len({r['physical_sample_id'] for r in rows}) if qualified else None,
            independent_source_groups=len({r['parent_source_group_id'] for r in rows}) if qualified else None,
            scope='qualified human-labelled development validation' if qualified else 'unconfirmed candidate-label diagnostic counts; NOT accuracy')
        if not qualified:result.update(candidate_top1_matches=correct,candidate_top3_contains=top3)
        for key,v in per_class.items():
            if qualified:
                v.update(top1=v['top1_matches']/v['images'] if v['images'] else None,top3=v['top3_contains']/v['images'] if v['images'] else None,
                         physical_objects=len({r['physical_sample_id'] for r in rows if r['candidate_class_key']==key}),
                         independent_source_groups=len({r['parent_source_group_id'] for r in rows if r['candidate_class_key']==key}))
        return result
    return {model:dict(development=metrics([r for r in records if r['route']=='qualified_dev'],model,True),
                       diagnostic=metrics([r for r in records if r['route']=='diagnostic_only'],model,False)) for model in ['v2','R1']}


def contact_sheets(rows, source, out, *, predictions=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    font_path=Path('C:/Windows/Fonts/msyh.ttc');font=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
    index=[]
    for start in range(0,len(rows),16):
        batch=rows[start:start+16];canvas=Image.new('RGB',(1600,((len(batch)+3)//4)*290+35),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,6),'Candidate labels only; no independent identity or human truth inferred',font=font,fill='black')
        for slot,row in enumerate(batch):
            x=slot%4*400;y=slot//4*290+35
            if row['decode_status']=='OK':
                with Image.open(safe_child(source,row['relative_path'])) as image:
                    tile=ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'),(394,218));canvas.paste(tile,(x,y))
            lines=[f'{start+slot+1} {row["relative_path"]}',f'candidate={row["candidate_class_key"]} sha={row["sha256"][:10]}']
            if predictions:lines.append(f'v2={V2_KEYS[row["v2"]["top1_id"]]} | R1={V2_KEYS[row["R1"]["top1_id"]]}')
            for line,text in enumerate(lines):draw.text((x+3,y+220+20*line),text,font=font,fill='black')
            index.append(dict(image_id=row['image_id'],relative_path=row['relative_path'],page=f'page_{start//16+1:03d}.jpg',slot=slot+1))
        canvas.save(out/f'page_{start//16+1:03d}.jpg',quality=94)
    write_csv(out/'index.csv',index,fields=['image_id','relative_path','page','slot'])
    return index


def predict(audit_dir,out,plan_sha,v2_path,r1_path,device='cuda:0'):
    import torch
    from .monitor import model_state_sha256,tensor_sha256
    audit_dir=Path(audit_dir);plan_path=audit_dir/'pre_prediction_plan.json'
    if digest(plan_path)!=plan_sha:raise ValueError('Pre-prediction qualification plan SHA changed')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if digest(audit_dir/'inventory.json')!=plan['inventory_sha256'] or digest(audit_dir/'source_snapshot.json')!=plan['source_snapshot_sha256']:
        raise ValueError('Audit decisions or source snapshot changed')
    rows=json.loads((audit_dir/'inventory.json').read_text(encoding='utf-8'));source=Path(plan['source_root'])
    expected=json.loads((audit_dir/'source_snapshot.json').read_text(encoding='utf-8'))
    if {p.relative_to(source).as_posix():digest(safe_child(source,p.relative_to(source).as_posix())) for p in source.rglob('*') if p.is_file()}!=expected:
        raise ValueError('Source changed after qualification freeze')
    out=output_dir(out);out.mkdir(parents=True,exist_ok=False);configure_sdk(out/'sdk-settings')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    networks={};contract={}
    for name,path,sha in [('v2',v2_path,V2_SHA),('R1',r1_path,R1_SHA)]:
        model,metadata=load_model(path,device,sha)
        if metadata['class_keys']!=V2_KEYS or metadata['preprocess']!=PREPROCESS or metadata['class_list_sha256']!=class_digest(V2_KEYS):
            raise ValueError('Class order/preprocess contract mismatch')
        if any(module.training for module in model.modules()):raise ValueError('Frozen eval requires every module in eval mode')
        networks[name]=model;contract[name]=dict(checkpoint_sha256=sha,model_state_before=model_state_sha256(model),class_keys=V2_KEYS,
            class_list_sha256=metadata['class_list_sha256'],preprocess=PREPROCESS,head_shape=list(model.model[-1].linear.weight.shape))
    selected=[r for r in rows if r['route']!='sealed_excluded' and r['decode_status']=='OK'];predictions=[];preprocess=transform(False)
    with torch.inference_mode():
        for start in range(0,len(selected),16):
            members=selected[start:start+16];tensors=[]
            for row in members:
                path=safe_child(source,row['relative_path'])
                if digest(path)!=row['sha256']:raise ValueError('Original image changed')
                with Image.open(path) as image:tensors.append(preprocess(image))
            batch=torch.stack(tensors);input_hash=tensor_sha256(batch);device_batch=batch.to(device);outputs={}
            for name,model in networks.items():
                probabilities=model(device_batch)[0].detach().cpu()
                if tuple(probabilities.shape)!=(len(members),12) or not bool(torch.isfinite(probabilities).all()):raise ValueError('Invalid classifier output')
                outputs[name]=probabilities
            if tensor_sha256(device_batch)!=input_hash:raise ValueError('Predictor changed shared input tensor')
            for i,row in enumerate(members):
                record=dict(row,input_sha256=tensor_sha256(tensors[i]))
                for name,probabilities in outputs.items():
                    values=probabilities[i];order=values.argsort(descending=True,stable=True)[:3].tolist()
                    record[name]=dict(top1_id=order[0],top3_ids=order,scores=values.tolist(),top3=[dict(class_id=j,class_key=V2_KEYS[j],score=float(values[j])) for j in order])
                predictions.append(record)
    for name,model in networks.items():
        contract[name]['model_state_after']=model_state_sha256(model)
        if contract[name]['model_state_before']!=contract[name]['model_state_after']:raise ValueError('Frozen inference changed model state')
    write_json_new(out/'contracts.json',contract);write_json_new(out/'predictions.json',predictions)
    fields=['image_id','relative_path','sha256','input_sha256','candidate_class_key','class_id','declared_group','route','trusted_label',
            'independent_physical_count','physical_sample_id','session_id','parent_source_group_id','background','foil','shelf']
    flat=[]
    for record in predictions:
        row={k:record.get(k) for k in fields}
        for name in networks:row.update({name+'_top1':V2_KEYS[record[name]['top1_id']],name+'_top1_score':record[name]['scores'][record[name]['top1_id']],
                                        name+'_top3':json.dumps(record[name]['top3']),name+'_scores':json.dumps(record[name]['scores'])})
        flat.append(row)
    write_csv(out/'predictions.csv',flat,fields=fields+[f'{name}_{suffix}' for name in networks for suffix in ['top1','top1_score','top3','scores']])
    summary=score(predictions);write_json_new(out/'summary.json',summary)
    errors=[r for r in predictions if any(V2_KEYS[r[m]['top1_id']]!=r['candidate_class_key'] for m in networks)]
    contact_sheets(errors,source,out/'error_contacts',predictions=True)
    if digest(v2_path)!=V2_SHA or digest(r1_path)!=R1_SHA:raise ValueError('Checkpoint bytes changed')
    if digest(plan_path)!=plan_sha:raise ValueError('Qualification plan changed during inference')
    return predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    a=commands.add_parser('audit');a.add_argument('--source',type=Path,required=True);a.add_argument('--training-rows',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    p=commands.add_parser('predict');p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--v2',type=Path,required=True);p.add_argument('--r1',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.command=='audit':
        training=[r for r in json.loads(args.training_rows.read_text(encoding='utf-8')) if r['new_split']=='train']
        rows=audit(args.source,training,args.out);print(json.dumps(dict(images=len(rows),roles=dict(Counter(r['route'] for r in rows)))))
    else:
        rows=predict(args.audit_dir,args.out,args.plan_sha,args.v2,args.r1,args.device);print(json.dumps(dict(predicted=len(rows),formal_accuracy_requires_qualified_dev=True)))


if __name__=='__main__':main()
