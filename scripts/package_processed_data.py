"""Package ALL processed files for GitHub release delivery, with byte verification.

ZIP paths are repository-relative. The training bundle additionally contains
data/manifest.csv so extracting it at the repository root restores training paths.
This script only creates local archives; authentication/upload is separate.
"""
import hashlib
import json
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
    DESTINATION.mkdir(parents=True, exist_ok=True)
    processed = ROOT / 'data/processed'
    actual_folders = {p.name for p in processed.iterdir() if p.is_dir()}
    if actual_folders != set(BUNDLES) or any(p.is_file() for p in processed.iterdir()):
        raise ValueError('Processed contents changed: define bundles for every item before publishing')
    index = []
    packages = []
    for folder, filename in BUNDLES.items():
        paths = sorted(p for p in (processed / folder).rglob('*') if p.is_file())
        source_paths = set(paths)
        if folder == 'selected_dataset_v1':
            paths.append(ROOT / 'data/manifest.csv')
        expected = {}
        print(f'Packaging {folder}: {len(paths)} files', flush=True)
        archive_path = DESTINATION / filename
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for number, path in enumerate(paths):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f'Unsafe source: {path}')
                relative = path.relative_to(ROOT).as_posix()
                with path.open('rb') as stream:
                    value = digest(stream)
                expected[relative] = value
                archive.write(path, relative)
                if path in source_paths:
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
    source_set = {p.relative_to(ROOT).as_posix() for p in processed.rglob('*') if p.is_file()}
    if {entry['path'] for entry in index} != source_set or len(index) != len(source_set):
        raise ValueError('Processed-directory coverage is not exactly complete')
    report = {'processed_files': len(index), 'processed_bytes': sum(e['bytes'] for e in index),
              'coverage': 'all_processed_files', 'packages': packages, 'files': index}
    (DESTINATION / 'processed-file-inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (DESTINATION / 'package_summary.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
