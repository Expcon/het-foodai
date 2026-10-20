"""Create a local HTML QA report from actual audit output, without network calls."""
import argparse
from html import escape
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path(__file__).resolve().parents[1] / 'data/cache/audit_v1')
    args = parser.parse_args()
    root = args.audit.resolve()
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    names = {'corn': '玉米', 'potato': '土豆', 'broccoli': '西兰花', 'pumpkin': '南瓜',
             'chicken_wing': '鸡翅', 'chicken_breast': '鸡胸肉',
             'sea_bass_unverified': '泛鱼类（不能确认为鲈鱼）',
             'shrimp_unverified': '泛虾类（对虾身份未确认）'}
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>食材数据集审核报告</title><style>',
            'body{max-width:1120px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#20312b;background:#f7faf8}',
            'table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d8e2dc;padding:9px;text-align:left}',
            'img{max-width:100%;height:auto}section{margin:28px 0;padding:20px;background:white;border:1px solid #d8e2dc}',
            'a{color:#12664c}.notice{padding:16px;background:#fff1d4}code{overflow-wrap:anywhere}</style>',
            '<h1>食材数据集审核报告</h1>',
            '<p class="notice">本报告是训练前数据质量检查。候选图尚未人工接纳，未冻结类别或训练/验证/测试划分。数量不能解释为独立拍摄样本量。</p>',
            '<h2>来源文件检查</h2><table><tr><th>来源</th><th>图片数</th><th>原包 SHA256</th></tr>']
    for package in summary['packages']:
        body.append(f'<tr><td>{escape(package["source_id"])}</td><td>{package["images"]:,}</td>'
                    f'<td><code>{escape(package["sha256"])}</code></td></tr>')
    body.append('</table><p>解码结果：' + escape(str(summary['image_status'])) +
                f'；问题记录 {summary["issues"]:,} 条。框异常与语义问题不等于文件损坏。</p>')
    body.append('<h2>候选样本盘点</h2><table><tr><th>类别</th><th>图片/裁剪数</th><th>不同解码像素数</th><th>来源父图数</th></tr>')
    for key, count in sorted(summary['candidate_counts'].items()):
        body.append(f'<tr><td>{escape(names.get(key, key))}</td><td>{count:,}</td>'
                    f'<td>{summary["candidate_unique_pixels"][key]:,}</td><td>{summary["candidate_parent_images"][key]:,}</td></tr>')
    body.append('</table><p>不同像素仍可能来自同一拍摄、增强或近重复，不能称作独立原图。泛鱼与泛虾仅供复核，不计入鲈鱼/对虾正式数据。</p>')
    body.append('<p class="notice">视觉抽查发现部分 Roboflow 图片有严重噪点、黑块遮挡和低分辨率；corn 中也有罐头/容器样本。全部框数不能直接当作合格食材照片数量。鲈鱼来源和对虾语义仍未确认。</p>')
    body.append('<h2>重复与泄漏风险</h2><ul>')
    for field, label in [('sha256', '文件字节完全相同'), ('pixel_sha256', '解码 RGB 像素完全相同')]:
        body.append(f'<li>{label}：{summary["original_exact_group_counts"][field]:,} 组；'
                    f'多余副本 {summary["original_exact_excess_copies"][field]:,} 张；'
                    f'同来源跨原始划分 {summary["original_exact_cross_split_groups"][field]:,} 组。</li>')
    body.append('</ul><p>上述两种统计彼此重叠，不能相加。pHash 近重复是复核线索，有误报及漏报，未据此自动删除。</p>')
    pairs = summary['pair_counts']
    body.append(f'<p>原图近重复候选 {pairs.get("original_pairs", 0):,} 对；'
                f'其中同来源跨原始划分 {pairs.get("original_cross_original_split_pairs", 0):,} 对，'
                f'跨来源 {pairs.get("original_cross_source_pairs", 0):,} 对。'
                f'分类候选近重复 {pairs.get("candidate_pairs", 0):,} 对。相似对不是独立图片数量，也不是已确认重复数量。</p>')
    if (root / 'cross_split_spotcheck.jpg').exists():
        body.append('<p>三组Kaggle跨集样本经辅助视觉复核可见相同画面，来源划分存在近重复泄漏：</p><img src="cross_split_spotcheck.jpg" alt="跨原始划分的西兰花、土豆、南瓜相似画面对比">')
    if (root / 'group_summary.json').exists():
        grouping = json.loads((root / 'group_summary.json').read_text(encoding='utf-8'))
        body.append(f'<p>保守分组草案：{grouping["candidate_samples"]:,} 个候选，'
                    f'{grouping["candidate_groups"]:,} 个关联组；最大组 {grouping["largest_candidate_group_samples"]:,} 个候选。'
                    '相似边可能误报或漏报，关联组尚未验证为独立拍摄组。</p>')
    body.append('<h2>审核文件</h2><ul>')
    for filename, label in [('summary.json', '完整统计与来源说明'), ('images.csv', '全部原图台账'),
                            ('human_review.csv', '逐图人工复核清单'), ('issues.csv', '异常记录'),
                            ('annotation_counts.csv', '检测标注数量'), ('original_exact_groups.json', '完全重复组'),
                            ('group_proposals.csv', '待复核关联组'), ('group_summary.json', '关联组摘要'),
                            ('original_near_pairs.csv', '原图近重复候选'), ('candidate_near_pairs.csv', '分类候选近重复'),
                            ('contact_index.csv', '抽查图格子索引')]:
        body.append(f'<li><a href="{filename}">{label}</a></li>')
    body.append('</ul><h2>分类抽查图册</h2><p>每类最多 24 张，按来源覆盖、相同像素去重。抽查不能代替全量标签验收。</p>')
    for path in sorted(root.glob('contact_*.jpg')):
        key = path.stem.removeprefix('contact_')
        body.append(f'<section><h3>{escape(names.get(key, key))}</h3>'
                    f'<a href="{escape(path.name)}"><img src="{escape(path.name)}" alt="{escape(key)} 分类抽查图"></a></section>')
    body.append('<p>下一步：复核标签与近重复组，补齐鲈鱼来源和对虾定义，再拟定关联组划分并履行共同冻结流程。</p></html>')
    path = root / 'report.html'
    path.write_text('\n'.join(body), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
