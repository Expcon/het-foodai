"""Synthetic audit fixtures only; none are real food examples."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, PngImagePlugin

from tools.oven_training.round2_audit import (
    CLASS_KEYS, FOLDER_CLASSES, audit_directory, group_rows, label_evidence,
    safe_child, scene_evidence,
)


class OvenRound2AuditTests(unittest.TestCase):
    def test_class_ids_append_to_v1_without_reordering(self):
        self.assertEqual(CLASS_KEYS, ('asparagus', 'bacon', 'broccoli', 'chicken_wing',
                                     'croissant', 'french_fries', 'potato', 'salmon',
                                     'toast', 'corn', 'sausage', 'steak'))
        self.assertEqual(FOLDER_CLASSES['牛排'], 'steak')
        self.assertEqual(len(FOLDER_CLASSES), 12)

    def test_pending_sidecar_and_wrong_candidate_never_become_truth(self):
        pending = {'label_status': 'pending_manual_review', 'camera_context': {
            'human_label_confirmed': False, 'candidate_class_key_unreviewed': 'pumpkin'}}
        result = label_evidence('croissant', '可颂/a.jpg', pending, False)
        self.assertFalse(result['trusted_label'])
        self.assertEqual(result['label_status'], 'candidate_directory')
        pending['camera_context']['human_label_confirmed'] = True
        self.assertFalse(label_evidence('croissant', '可颂/a.jpg', pending, False)['trusted_label'])
        photo = label_evidence('toast', '吐司/photo mid_0001.png', pending, True)
        self.assertTrue(photo['trusted_label'])
        self.assertEqual(photo['label_status'], 'human_reviewed_from_organizer')
        self.assertFalse(label_evidence('toast', '吐司/photo other_0001.png', pending, True)['trusted_label'])

    def test_ambiguous_have_no_and_scene_ids_remain_uncertain(self):
        data = scene_evidence({'sample_prefix': 'beef_have_have_001', 'camera_context': {
            'physical_sample_id': 'beef_have_have_001', 'session_id': 'synthetic-session'}}, 'a.jpg')
        self.assertEqual(data['foil']['value'], 'unknown')
        self.assertEqual(data['shelf']['value'], 'unknown')
        self.assertEqual(data['oven_id'], 'unknown')
        self.assertFalse(data['physical_identity_verified'])
        explicit = scene_evidence({'sample_prefix': 'salmon-high-without_foil'}, 'a.jpg')
        self.assertEqual(explicit['foil']['value'], 'absent')
        self.assertEqual(explicit['shelf']['value'], 'high')

    def test_conflicting_filename_and_sidecar_shelf_is_unknown(self):
        result = scene_evidence({'sample_prefix': 'chips_a_mid_have'}, 'chips_a_high_have_0001.png')
        self.assertEqual(result['shelf']['value'], 'unknown')
        self.assertIn('conflict', result['shelf']['evidence'])

    def test_group_graph_joins_session_duplicates_and_unknown_same_class(self):
        def row(i, label, source, sha, pixel, session='unknown', physical='unknown'):
            return {'image_id': i, 'class_key': label, 'source': source, 'sha256': sha,
                    'pixel_sha256': pixel, 'session_id': session, 'physical_sample_id': physical,
                    'status': 'ok', 'trusted_label': False, 'relative_path': i,
                    'physical_identity_verified': False}
        rows = [row('a', 'toast', 'new', '1', 'p1', 's1', 'scene1'),
                row('b', 'toast', 'round1', '2', 'p2'),
                row('c', 'bacon', 'new', '3', 'p3', 's1', 'scene2'),
                row('d', 'corn', 'new', '4', 'p4', 's2', 'scene3'),
                row('e', 'corn', 'new', '5', 'p4', 's2', 'scene3')]
        group_rows(rows, [{'left': 'c', 'right': 'd'}])
        self.assertEqual(len({r['group_id'] for r in rows}), 1)
        self.assertEqual(rows[-1]['new_split'], 'excluded')
        self.assertEqual(rows[-1]['duplicate_of'], 'd')
        self.assertEqual(rows[-1]['exclusion_reason'], 'duplicate_pixels')
        self.assertTrue(all(r['new_split'] in {'train', 'excluded'} for r in rows))

    def test_conflicting_pixel_labels_exclude_both(self):
        rows = [{'image_id': key, 'class_key': key, 'source': 'new', 'sha256': key,
                 'pixel_sha256': 'same', 'status': 'ok', 'relative_path': key}
                for key in ['toast', 'corn']]
        group_rows(rows, [])
        self.assertTrue(all(r['exclusion_reason'] == 'conflicting_labels_for_identical_pixels' for r in rows))

    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['../outside.png', 'C:/outside.png', '/outside.png', '..\\outside.png']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_child(Path(directory), name)

    def test_real_decode_and_index_hashes_use_read_only_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / 'synthetic_input'
            folder = source / '吐司'
            folder.mkdir(parents=True)
            image = Image.new('RGB', (19, 13), (31, 91, 141))
            image.save(folder / 'a.png')
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('fixture', 'synthetic encoding difference')
            image.save(folder / 'b.png', pnginfo=metadata)
            (folder / 'broken.png').write_bytes(b'synthetic broken image')
            records = []
            for file in sorted(folder.iterdir()):
                records.append({'category': '吐司', 'file_type': 'image', 'source': 'synthetic/' + file.name,
                                'destination': file.relative_to(source).as_posix(),
                                'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                'size_bytes': file.stat().st_size, 'classification_basis': 'synthetic'})
            sidecar = folder / 'a.json'
            sidecar.write_text(json.dumps({'image_file': 'historical_name.png',
                'sha256': records[0]['sha256'], 'width': 19, 'height': 13,
                'label_status': 'pending_manual_review'}), encoding='utf-8')
            records.append({'category': '吐司', 'file_type': 'metadata', 'source': 'synthetic/a.json',
                            'destination': '吐司/a.json', 'sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                            'size_bytes': sidecar.stat().st_size, 'classification_basis': 'synthetic'})
            with (source / '分类清单.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            (source / '分类核验.json').write_text(json.dumps({'image_count': 3, 'class_count': 1,
                                                        'metadata_count': 0, 'image_counts': {'吐司': 3}}), encoding='utf-8')
            (source / '整理说明.md').write_text('Synthetic fixture, no reviewed labels.', encoding='utf-8')
            before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob('*') if p.is_file()}
            summary, rows = audit_directory(source, repo, old_rows_path=None, phash_threshold=0)
            self.assertEqual(summary['new']['image_count'], 3)
            self.assertEqual(summary['new']['image_status_counts'], {'decode_error': 1, 'ok': 2})
            self.assertEqual(summary['index']['sha_mismatch_count'], 0)
            self.assertEqual(summary['duplicates']['new_new']['pixel_sha256']['excess_copies'], 1)
            self.assertEqual(sum(r['new_split'] == 'train' for r in rows), 1)
            first = next(r for r in rows if r['relative_path'] == '吐司/a.png')
            self.assertEqual(first['new_split'], 'train')
            self.assertEqual(first['metadata_warnings'], ['sidecar_filename_differs_but_sha256_matches'])
            self.assertEqual(before, {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in source.rglob('*') if p.is_file()})
            self.assertTrue((repo / 'runs/oven-round2-20261107/audit/combined_rows.json').is_file())


if __name__ == '__main__':
    unittest.main()
