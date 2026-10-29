"""Package ALL processed files for GitHub release delivery, with byte verification.

ZIP paths are repository-relative. The training bundle additionally contains
data/manifest.csv so extracting it at the repository root restores training paths.
This script only creates local archives; authentication/upload is separate.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'data/cache/github_upload/processed_release_v1'
BUNDLES = {
    'selected_dataset_v1': 'het-foodai-selected-dataset-v1.zip',
    'audit_candidates_v1': 'het-foodai-audit-candidates-v1.zip',
}


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-version', help='Package only this selected_dataset_vN; retain older published versions')
    args = parser.parse_args()
    destination = DESTINATION
    bundles = BUNDLES
    if args.selected_version:
        version = args.selected_version
        if not re.fullmatch(r'selected_dataset_v\d+(?:\.\d+)*', version):
            raise ValueError('Expected selected_dataset_vN or selected_dataset_vN.N.N')
        destination = ROOT / 'data/cache/github_upload' / version.replace('selected_dataset_', 'processed_release_')
        bundles = {version: 'het-foodai-selected-dataset-' + version.removeprefix('selected_dataset_') + '.zip'}
    destination.mkdir(parents=True, exist_ok=True)
    processed = ROOT / 'data/processed'
    actual_folders = {p.name for p in processed.iterdir() if p.is_dir()}
    if not args.selected_version and (actual_folders != set(bundles) or any(p.is_file() for p in processed.iterdir())):
        raise ValueError('Processed contents changed: define bundles for every item before publishing')
    index = []
    packages = []
    for folder, filename in bundles.items():
        if not (processed / folder).is_dir():
            raise ValueError('Selected dataset folder does not exist')
        source_paths = set(p for p in (processed / folder).rglob('*') if p.is_file())
        paths = {p.relative_to(ROOT).as_posix(): p for p in sorted(source_paths)}
        if folder.startswith('selected_dataset_'):
            if not args.selected_version and (processed / folder / 'manifest.csv').read_bytes() != (ROOT / 'data/manifest.csv').read_bytes():
                raise ValueError('Root manifest does not match selected dataset')
            paths['data/manifest.csv'] = processed / folder / 'manifest.csv'
        expected = {}
        print(f'Packaging {folder}: {len(paths)} files', flush=True)
        archive_path = destination / filename
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for number, (relative, path) in enumerate(paths.items()):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f'Unsafe source: {path}')
                with path.open('rb') as stream:
                    value = digest(stream)
                expected[relative] = value
                archive.write(path, relative)
                if relative != 'data/manifest.csv':
                    index.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': value, 'archive': filename})
                if (number + 1) % 2000 == 0:
                    print(f'  {number + 1}/{len(paths)} archived', flush=True)
        # Read every member: both ZIP CRC and source-byte SHA must pass.
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError('Archive omitted or duplicated a source file')
            for member in archive.namelist():
                with archive.open(member) as stream:
                    if digest(stream) != expected[member]:
                        raise ValueError(f'Archive-byte verification failed: {member}')
        with archive_path.open('rb') as stream:
            archive_digest = digest(stream)
        if archive_path.stat().st_size >= 2 * 1024 ** 3:
            raise ValueError('Archive exceeds the GitHub release per-asset limit; split it first')
        packages.append({'name': filename, 'bytes': archive_path.stat().st_size, 'sha256': archive_digest,
                         'processed_files': len(source_paths), 'total_members': len(paths)})
        print(f'Verified {filename}: {archive_path.stat().st_size:,} bytes', flush=True)
    scope = processed / args.selected_version if args.selected_version else processed
    source_set = {p.relative_to(ROOT).as_posix() for p in scope.rglob('*') if p.is_file()}
    if {entry['path'] for entry in index} != source_set or len(index) != len(source_set):
        raise ValueError('Processed-directory coverage is not exactly complete')
    report = {'processed_files': len(index), 'processed_bytes': sum(e['bytes'] for e in index),
              'coverage': args.selected_version or 'all_processed_files', 'packages': packages, 'files': index}
    (destination / 'processed-file-inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'package_summary.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
