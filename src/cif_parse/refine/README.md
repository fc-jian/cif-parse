# Antibody-Antigen Complex Refinement

对 cif-parse 识别出的 antibody-antigen complex 进行结构裁剪：保留抗体 Fv 区域，通过 residue-level contact graph + Louvain community detection 筛选与抗体接触的抗原 domain。

## 快速开始

```bash
cif-parse-refine abag \
    --case-dirs ./outputs/cases/ \
    --prep-dir ./prep \
    --outdir ./refined
```

## 输出

每个 complex 产生两个文件：

| 文件 | 说明 |
|------|------|
| `{pdb}_{assembly}_{complex_id}_refined.pdb` | 裁剪后的多链 PDB，chain 重新编号为 A/B/C... |
| `{pdb}_{assembly}_{complex_id}_refined.json` | 元数据（见下方 schema） |

### JSON Schema

```json
{
  "pdb_id": "5ywo",
  "assembly_id": "2",
  "complex_id": "ab_ag_001",
  "source_path": ".../5ywo.cif.gz",
  "antibody_unit_type": "paired_heavy_light",
  "chain_intervals": [
    {
      "label_asym_id": "H",
      "pdb_chain_id": "A",
      "chain_type": "antibody heavy chain",
      "role": "antibody",
      "retained_residue_intervals": [[1, 118]]
    },
    {
      "label_asym_id": "G",
      "pdb_chain_id": "B",
      "chain_type": "antibody light chain",
      "role": "antibody",
      "retained_residue_intervals": [[1, 108]]
    },
    {
      "label_asym_id": "C",
      "pdb_chain_id": "C",
      "chain_type": "other protein chain",
      "role": "antigen",
      "retained_residue_intervals": [[100, 153], [200, 244], [350, 399]]
    }
  ],
  "antigen_domains": [
    {"label_asym_id": "C", "num_residues": 54, "num_antibody_contacts": 3}
  ],
  "removed_antigen_domains": [...],
  "contact_summary": {"distance_threshold": 8.0, "louvain_resolution": 1.0}
}
```

- `retained_residue_intervals`: 紧凑区间表示 `[[start, end], ...]`，使用原始 PDB `label_seq_id`
- `pdb_chain_id`: 输出 PDB 中的 chain ID（A-Z, a-z, 0-9）
- `label_asym_id`: 原始 mmCIF 中的 chain 标识

## 算法

1. **抗体 Fv 裁剪**: 使用 SADIE IMGT 注释的 `variable_domains`，对 paired Fab 裁剪到 Fv 区域（VHH/scFv 完整保留）
2. **抗原 domain 鉴定**: 计算 Cα-Cα 距离（≤ 8.0 Å）建立残基邻接图 → Louvain community detection 识别 domain
3. **Domain 筛选**: 保留与抗体 Fv 有 ≥ 3 个残基接触的 domain，删除其他

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--contact-distance` | 8.0 | 残基接触距离阈值 (Å) |
| `--louvain-resolution` | 1.0 | Louvain 分辨率参数 |
| `--min-domain-size` | 10 | 最小 domain 残基数 |
| `--min-contact-residues` | 3 | domain 保留的抗体接触残基数 |

## 依赖

- `biotite` — atom array 操作
- `networkx` — 图构建和 Louvain community detection
- `numpy` — 距离计算
- `cif_parse.clustering.prep` — 读取 chain atom arrays

## 可视化（可选）

`render_comparison.py` 是一个独立的 PyMOL 可视化脚本，生成 refinement 前后对比图。*不增加 PyMOL 为项目依赖*，需要时单独安装 `pymol-open-source`。

```bash
# 安装 PyMOL（一次性）
pip install pymol-open-source

# 生成对比图
python -m cif_parse.refine.render_comparison \
    --case-dir ./outputs/cases/5ywo \
    --prep-dir ./prep \
    --assembly-id 2 \
    --complex-id ab_ag_001 \
    --outdir ./comparison_images
```

输出：
- `ab_ag_001_comparison.pse` — PyMOL session，同时包含 before/after 两个对象，可在 PyMOL 中交互查看
- `ab_ag_001_before_full.png` — 完整 Fab + 抗原（所有 domain）
- `ab_ag_001_after_refined.png` — Fv 区域 + 仅抗体接触的抗原 domain
