"""Conservative review groups; no split or training manifest is produced.

Same-parent, exact/pixel duplicate and pHash candidate edges are joined globally,
including across sources. pHash edges still require review before freezing.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


class UnionFind:
    def __init__(self, ids):
        self.parent = {item: item for item in ids}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def propose(root):
    images = read_csv(root / 'images.csv')
    samples = read_csv(root / 'human_review.csv')
    union = UnionFind(row['image_id'] for row in images)
    sample_parent = {r['image_id']: r['parent_image_id'] for r in samples}
    for field in ['sha256', 'pixel_sha256']:
        seen = {}
        for row in images:
            value = row.get(field)
            if value:
                if value in seen:
                    union.join(row['image_id'], seen[value])
                seen[value] = row['image_id']
    for filename, mapping in [('original_near_pairs.csv', None), ('candidate_near_pairs.csv', sample_parent)]:
        with (root / filename).open(encoding='utf-8-sig', newline='') as stream:
            for pair in csv.DictReader(stream):
                left, right = pair['left'], pair['right']
                if mapping is not None:
                    left, right = mapping[left], mapping[right]
                union.join(left, right)
    components = defaultdict(list)
    for row in images:
        components[union.find(row['image_id'])].append(row['image_id'])
    group_ids = {key: 'proposal_' + hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:24]
                 for key, members in components.items()}
    output = []
    by_class = defaultdict(set)
    group_sizes = defaultdict(int)
    for row in samples:
        group = group_ids[union.find(row['parent_image_id'])]
        output.append({'image_id': row['image_id'], 'parent_image_id': row['parent_image_id'],
                       'source_id': row['source_id'], 'class_key': row['class_key'], 'proposed_group_id': group,
                       'review_status': row['review_status'], 'group_status': 'requires_review_not_independent_count'})
        by_class[row['class_key']].add(group)
        group_sizes[group] += 1
    with (root / 'group_proposals.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id', 'parent_image_id', 'source_id', 'class_key',
                                                   'proposed_group_id', 'review_status', 'group_status'])
        writer.writeheader()
        writer.writerows(output)
    summary = {'status': 'conservative_candidate_groups_pending_review',
               'candidate_samples': len(samples), 'candidate_groups': len(group_sizes),
               'class_group_counts': {key: len(values) for key, values in sorted(by_class.items())},
               'largest_candidate_group_samples': max(group_sizes.values(), default=0),
               'limitations': ['All pHash candidate edges joined conservatively, including possible false positives.',
                               'Groups are not verified independent capture groups.',
                               'No final split, classes freeze or formal manifest produced.']}
    (root / 'group_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path(__file__).resolve().parents[1] / 'data/cache/audit_v1')
    propose(parser.parse_args().audit.resolve())
