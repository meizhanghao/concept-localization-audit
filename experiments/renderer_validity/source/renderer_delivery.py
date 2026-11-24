"""Package the renderer follow-up and write its local scientific report."""
from .common import OUT
from .renderer_validity import RUN, FB, MODELS, digest
from pathlib import Path
import json
import shutil
import numpy as np


def number(value):
    return '未定义' if value is None else f'{value:.4f}'


def ci(record):
    return ('未定义' if record['ci95'] is None else
            f"[{record['ci95'][0]:.4f}, {record['ci95'][1]:.4f}]")


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entry = result['cohorts']['complement300']['analyses']
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=180)
    colors = ['#346b9a', '#be6326']
    for k, support in enumerate(['all', 'visible']):
        points = [entry[support]['models'][model]['gradient']['concept_margin']['deltaG_minus_S']
                  for model, _, _ in MODELS]
        means = np.array([p['estimate'] for p in points])
        bounds = np.array([p['ci95'] for p in points])
        axes[0].errorbar(means, np.arange(3)+(k-.5)*.16,
                         xerr=np.stack([means-bounds[:, 0], bounds[:, 1]-means]),
                         fmt='o', ms=4, capsize=3, color=colors[k], label=support+' recipients')
    axes[0].set_yticks(np.arange(3), ['R18 s0', 'R18 s1', 'R50 s2'])
    axes[0].set_xlabel('Spearman difference: delta_g minus s')
    axes[0].set_xticks([-.10, -.05, 0, .05])
    axes[0].set_title('Gradient response association')
    for k, method in enumerate(['cam', 'gradient']):
        records = entry['all']['models']['resnet18_seed0'][method]['class_margin_selection']
        points = [records[name]['selected_minus_uniform'] for name in ['S', 'G', 'deltaG']]
        means = np.array([p['estimate'] for p in points]); bounds = np.array([p['ci95'] for p in points])
        axes[1].errorbar(means, np.arange(3)+(k-.5)*.16,
                         xerr=np.stack([means-bounds[:, 0], bounds[:, 1]-means]),
                         fmt='o', ms=4, capsize=3, color=colors[k], label=method)
    axes[1].set_yticks(np.arange(3), ['s', 'g', 'delta_g'])
    axes[1].set_xlabel('Selected minus uniform class-margin response')
    axes[1].set_title('R18 s0 part-selection utility')
    for ax in axes:
        ax.axvline(0, color='#777777', lw=.8, linestyle='--')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Exploratory 300-image follow-up; pointwise paired image intervals', fontsize=10)
    fig.tight_layout()
    fig.savefig(RUN/'renderer_criterion.png')
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    bundle = base/'renderer_bundle'
    data, source = bundle/'data', bundle/'source'
    data.mkdir(exist_ok=True); source.mkdir(exist_ok=True)
    for original, destination in [('inputs.npz', 'inputs.npz'), ('results.json', 'expected.json'),
                                   ('protocol.json', 'protocol.json'), ('inference_checks.json', 'inference_checks.json'),
                                   ('analysis_amendment.json', 'analysis_amendment.json'),
                                   ('analysis_checks.json', 'analysis_checks.json')]:
        shutil.copyfile(RUN/original, data/destination)
    shutil.copyfile(base/'renderer_statistics.py', bundle/'renderer_statistics.py')
    for name in ['renderer_validity.py', 'renderer_delivery.py', 'common.py', 'models.py', 'funnybirds.py']:
        shutil.copyfile(base/name, source/name)
    provenance = {'checkpoints': {name: digest(FB/name) for _, name, _ in MODELS},
                  'caches': {name: digest(FB/name) for name in ['test_rgb.npy', 'deleted_rgb.npy',
                              'test_masks.npy', 'metadata.npz', 'parts.json', 'readouts.pt', 'prepared.json']},
                  'sources': {name: digest(base/name) for name in ['renderer_validity.py', 'renderer_statistics.py']}}
    (data/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    manifest = {p.relative_to(bundle).as_posix(): digest(p) for p in sorted(bundle.rglob('*'))
                if p.is_file() and p.name != 'manifest.json'}
    (bundle/'manifest.json').write_text(json.dumps({'scope': 'Exploratory renderer criterion follow-up, separate from base/frontier inputs.',
                                                   'files': manifest}, indent=2)+'\n', encoding='utf-8')
    result = json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    plot(result)
    follow = result['cohorts']['complement300']
    lines = ['# 渲染干预效度扩展：结果与论文取舍', '',
             '本轮仅扩展 A；B 保持训练压力测试，C 保持有限参照 bank 下的 abstain，D 保持报告中的前提诊断。没有把常用删除、相关或 regret 包装成新方法。', '',
             '## 先纠正事实与理论', '',
             '- 旧 A 已经是 FunnyBirds，而不是 CUB；使用官方同场景部件删除，且已经计算了 uniform/oracle regret。',
             '- 当前稿在本轮开始已为 14 页，A 段落、Table 6 的 S 行、3 Gaussian + 1 channel 和宽度六配对说明都已存在。',
             '- 明确的干预是“固定原场景，删除一个部件”；结果是固定概念对比 logit 的变化。其样本平均可以称为该渲染操作下的平均模型响应，不自动成为现实世界语义 ACE。CUB 补全也不会自动解决这点。',
             '- S 或 ΔG 优化后，删除响应区间跨零表示改善未决，不能证明无改善、等效或 gaming 成立。', '',
             '## 设计与执行', '',
             '三个现有检查点：R18 s0、R18 s1、R50 s2；原 200 图用于实现复核，另 300 图用于探索性扩展。后 300 图早已参与布局校准，不能称为独立确认。共 500 图、每图五个查询，三个模型；没有新训练或挑选最佳种子。', '',
             '每个解释器使用三个 row-norm matched Gaussian 头、200 个 exact query/alternative 供体排列。r=24；可见性只限制 recipient，不改变 donor。300 图中各部位可见查询数：'+str(follow['visible_queries_by_part'])+'。像素完全不变的删除数：'+str(follow['exactly_unchanged_deletions_by_part'])+'。这些部位均标为存在；不可见不等于不存在。', '',
             '原图和五个删除变体在同一前向批次计算；所有像素完全不变的变体的 logit 最大差为 0。原 200 图的三个训练模型峰值和 s0 三个随机头峰值与既有缓存逐点一致。', '',
             '## 300 图的主结果', '',
             '| 模型 / 方法 | 全部查询 ρ(S) | 全部查询 ρ(δg) | 差 δg−S | 可见查询差 δg−S | CAM 有定义 bootstrap 比例 |',
             '|---|---:|---:|---:|---:|---:|']
    for model, _, _ in MODELS:
        for method in ['cam', 'gradient']:
            a = follow['analyses']['all']['models'][model][method]['concept_margin']
            v = follow['analyses']['visible']['models'][model][method]['concept_margin']
            lines.append(f"| {model} / {method} | {number(a['S']['estimate'])} | {number(a['deltaG']['estimate'])} | {number(a['deltaG_minus_S']['estimate'])} | {number(v['deltaG_minus_S']['estimate'])} | {a['deltaG_minus_S']['valid_bootstrap_fraction']:.2%} |")
    lines += ['', 'S 对 CAM 的点估计关联在三个模型中均较高，但不能把它写成普适因果代理。梯度差值在可见查询上都转为小正点估计；这是不同条件总体，不能把其变化解释成“不可见因素的因果分解”。', '',
              '### 梯度：配对图像区间（所有抽样均有定义）', '',
              '| 模型 | 全部查询差与区间 | 可见查询差与区间 |', '|---|---|---|']
    for model, _, _ in MODELS:
        a = follow['analyses']['all']['models'][model]['gradient']['concept_margin']['deltaG_minus_S']
        v = follow['analyses']['visible']['models'][model]['gradient']['concept_margin']['deltaG_minus_S']
        lines.append(f"| {model} | {number(a['estimate'])} {ci(a)} | {number(v['estimate'])} {ci(v)} |")
    lines += ['', '这些为探索性 pointwise 区间，不是所有次要分析的同时覆盖保证。', '',
              '## 发现并修正的统计缺口', '',
              '旧 frontier_criterion.py 用 nanmean 平均抽样中仍有定义的部位。CAM 在部分部位命中恒定时，这会改变“固定五部位平均”这个统计量。新分析不静默删部位：该抽样的宏平均记为无定义，并报告有定义比例。原 200 图 R18 s1 CAM 的点估计也因一个部位恒定而无定义。', '',
              '因此 CAM 的 defined-draw 区间只是条件诊断，不称为无条件 95% CI；原计划六比较的 simultaneous 推断不具备完整定义，已撤销相应确认性结论。data/analysis_amendment.json 如实记录发现问题后的修正。旧 frontier 文件用于历史重算保留，但正文不再使用其可变部位数 CAM 区间。', '',
              '## 选择效用与 regret（次要，只限 R18 s0 冻结类别读出）', '',
              '每图选一个部位；分数最大值并列时均匀平均。响应为真实类相对原最强竞争类的类别 margin 删除变化。概念热图与类别效用目标不同，不能拿此项单独证伪概念解释。', '',
              '| 方法 / 分数 | 超过均匀选择的响应 | 95% 配对图像区间 | oracle regret |', '|---|---:|---|---:|']
    for method in ['cam', 'gradient']:
        values = follow['analyses']['all']['models']['resnet18_seed0'][method]
        for score in ['S', 'G', 'deltaG']:
            item = values['class_margin_selection'][score]
            lines.append(f"| {method} / {score} | {number(item['selected_minus_uniform']['estimate'])} | {ci(item['selected_minus_uniform'])} | {number(item['regret']['estimate'])} |")
    lines += ['', '所有效用均为事后诊断，不是人类研究或新算法；不以挑出一种分数的正结果作为门槛。完整可见性条件结果、各部位结果、配对 δg−S 选择差值在 results.json 中。', '',
              '## 与既有工作的关系和论文取舍', '',
              '- A：部件删除已有 FunnyBirds，模型干预与语义因果的区分已有 Cheng et al., Linking Model Intervention to Causal Interpretation in Model Explanation（本地论文已核读摘要）。这里的价值是审查 crossed score 与独立指定响应是否一致，并给出可见性/统计退化边界，不主张新 ACE 方法。',
              '- B：explanation supervision、mask supervision、有限预算压力测试已有相关工作。单种子不能改称训练方法；保留 caution。',
              '- C：finite-bank gate 不等于 stochastic dominance。共同参照在方法差值中消去；随机 map、平滑 map、改变 donor 的控制也未必保持同一 estimand。本轮不把这些强行拼成新理论。',
              '- D：输入非线性交互与线性读出交互不是同一个量；不为零的原图交互不能支持新的 concept coalition 指标。保留报告，不加入主文。', '',
              '主文用一段紧凑的 renderer criterion 结果替换旧 A 段落，重点是“不能把 correspondence 当作普适 response ranker”；保留 Table 6 的双评分改善与未决删除响应。完整 3 模型/2 条件结果随代码归档，不将 BlockSys 稿变成新的因果解释器论文。', '',
              '## 复现状态', '',
              f"数值包包含 {len(manifest)+1} 个文件（含自身 manifest），{len(manifest)} 条 SHA-256。只含峰、窗口、预测及读出参数，不含原图和编码器。发布目录为 github_cun/main/experiments/renderer_validity。离线期间只同步本地并提交，不重试联网，不把待推送提交写成已公开。", '']
    (RUN/'实验结果与论文取舍.md').write_text('\n'.join(lines), encoding='utf-8')
    print(bundle, len(manifest)+1, 'files')


if __name__ == '__main__':
    main()
