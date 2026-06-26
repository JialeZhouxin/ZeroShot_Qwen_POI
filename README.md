# ZeroShot_Qwen — LLM Zero-Shot Next POI Recommendation

基于 Qwen-2.5-7B-Instruct 的零样本下一兴趣点推荐评估。

## 快速开始

```bash
# 1. 安装依赖
uv venv && uv pip install -r requirements.txt

# 2. 准备数据（二选一）
python prepare_data.py                     # 方式 A: 从 data.zip 解压
python prepare_data.py --source <path>     # 方式 B: 从 processed_sthgcn data 生成

# 3. 设置 API Key
export OPENROUTER_API_KEY="sk-or-v1-..."

# 4. 运行
bash run.sh nyc     # NYC 412 样本, ~20 min
bash run.sh ca      # CA  711 样本, ~50 min
bash run.sh tky     # TKY 1890 样本, ~96 min
```

## 数据获取

### 方式 A：下载 data.zip

从以下链接下载 `data.zip`，放到项目根目录下，运行 `python prepare_data.py`：

- [百度网盘链接]（请补充）
- [Google Drive 链接]（请补充）

### 方式 B：从 processed_sthgcn data 生成

如果你已有 `processed_sthgcn data` 目录（含 `{NYC,TKY,CA}/sample.csv`）：

```bash
python prepare_data.py --source "path/to/processed_sthgcn data"
```

## 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | `qwen/qwen-2.5-7b-instruct` (OpenRouter API) |
| 推理方式 | 迭代猜测（10 选 1，猜错移除，最大 10 轮） |
| Temperature | 0 |
| 候选池大小 | 10（1 GT + 9 负样本） |
| 轨迹最小长度 | 5 |
| 评估指标 | Acc@1/5/10, NDCG@5/10, MRR, ValidRatio, MeanRank |

## 结果

| 数据集 | 样本数 | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG@5 | NDCG@10 |
|--------|--------|-------|-------|--------|-----|--------|---------|
| NYC | 412 | 0.3592 | 0.8544 | 1.0000 | 0.5600 | 0.6188 | 0.6657 |
| CA | 711 | 0.3319 | 0.7623 | 0.9859 | 0.5144 | 0.5539 | 0.6255 |
| TKY | 1890 | 0.3910 | 0.8619 | 0.9952 | 0.5880 | 0.6430 | 0.6863 |

详见 `REPORT.md`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序 |
| `prepare_data.py` | 数据准备脚本 |
| `run.sh` | 一键运行 |
| `requirements.txt` | Python 依赖 |
| `REPORT.md` | 详细评估报告 |
| `output/*_results.json` | 详细评估结果 |

## 备注

- **断点续跑**：每个样本结果缓存到 `output/{dataset}/{traj_id}`，中断后重跑自动跳过已完成样本
- **调试模式**：`bash run.sh nyc --debug` 仅跑 1 个样本
- **自定义样本数**：`uv run python main.py -d nyc --cases 50` 跑 50 个样本
