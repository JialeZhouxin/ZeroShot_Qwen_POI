# ZeroShot Qwen-2.5-7B-Instruct — 评估报告

## 实验目标

以零样本 LLM Prompt 方式评测 Qwen-2.5-7B-Instruct 在下一兴趣点推荐（Next POI Recommendation）任务上的表现。

## 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | `qwen/qwen-2.5-7b-instruct` via OpenRouter API |
| 推理方式 | 迭代猜测（每次从候选池选 1 个 POI，猜错移除，最大 10 轮） |
| Temperature | 0（贪婪解码） |
| 候选池大小 | 10（1 GT + 9 随机负样本，按 trajectory_id 固定种子） |
| 轨迹最小长度 | 5 |
| Prompt 结构 | 4 段式：长期偏好 + 短时轨迹（含 geohash）+ 时空上下文 + 候选池 |
| 输出格式 | `{"predicted_poi_id": "..."}` |
| 评估指标 | Acc@K, NDCG@K, MRR, ValidRatio, MeanRank |

### Prompt 设计

已对齐：
- Prompt 段落结构、时间格式化（含序数词 `1st/2nd/3rd`）、geohash precision=6
- 长期偏好统计逻辑（Daily routine / Frequent / Occasional，max 6 类）
- 候选池格式 `[ID: XXXX | Category: YYYY]`
- 迭代猜测评估协议
- NYC 额外过滤（排除仅 1 个类别用户 → 412 样本）
- TKY Shift-JIS 编码修复
- 全角冒号 `：`

---

## 实验结果

### 三数据集汇总

| 指标 | NYC (412) | CA (711) | TKY (1890) |
|------|-----------|----------|------------|
| Acc@1 | 0.3592 | 0.3319 | 0.3910 |
| Acc@5 | 0.8544 | 0.7623 | 0.8619 |
| Acc@10 | 1.0000 | 0.9859 | 0.9952 |
| NDCG@5 | 0.6188 | 0.5539 | 0.6430 |
| NDCG@10 | 0.6657 | 0.6255 | 0.6863 |
| MRR | 0.5600 | 0.5144 | 0.5880 |
| ValidRatio | 1.0000 | 1.0000 | 1.0000 |
| MeanRank | 3.01 | 3.51 | 2.81 |

### NYC — 412 样本

| 指标 | 值 |
|------|-----|
| Acc@1 | 0.3592 (148/412) |
| Acc@5 | 0.8544 (352/412) |
| Acc@10 | 1.0000 (412/412) |
| NDCG@5 | 0.6188 |
| NDCG@10 | 0.6657 |
| MRR | 0.5600 |
| MeanRank | 3.01 |

**效率：** 1,243 次 LLM 调用 | 3.02 次/样本 | 100% early stop | 0 API errors

### CA — 711 样本

| 指标 | 值 |
|------|-----|
| Acc@1 | 0.3319 (236/711) |
| Acc@5 | 0.7623 (542/711) |
| Acc@10 | 0.9859 (701/711) |
| NDCG@5 | 0.5539 |
| NDCG@10 | 0.6255 |
| MRR | 0.5144 |
| MeanRank | 3.51 |

10 个样本因模型输出无效 ID 导致调用耗尽未能命中。**效率：** 2,567 次 LLM 调用 | 3.61 次/样本 | 98.6% early stop | 0 API errors

### TKY — 1890 样本

| 指标 | 值 |
|------|-----|
| Acc@1 | 0.3910 (739/1890) |
| Acc@5 | 0.8619 (1629/1890) |
| Acc@10 | 0.9952 (1881/1890) |
| NDCG@5 | 0.6430 |
| NDCG@10 | 0.6863 |
| MRR | 0.5880 |
| MeanRank | 2.81 |

**效率：** 5,384 次 LLM 调用 | 2.85 次/样本 | 99.5% early stop | 0 API errors

---

## 结果分析

### 1. ZeroShot 的基线能力

ZeroShot Acc@1 在 33-39% 之间，远超 random baseline（10%）。Qwen-2.5-7B-Instruct 具备：

- **POI 类别语义理解**：能区分不同 POI 类别的出行场景
- **时空常识**：理解时间段（morning/afternoon/evening/night）与出行模式的关系
- **文本推理能力**：能从长期偏好分布 + 短时轨迹 + 时空约束中推断

### 2. 数据集难度差异

| 维度 | NYC | CA | TKY |
|------|-----|-----|-----|
| POI 数量 | 4,937 | 9,670 | 7,821 |
| 样本数 | 412 | 711 | 1,890 |
| Acc@1 | 0.3592 | 0.3319 | 0.3910 |
| MRR | 0.5600 | 0.5144 | 0.5880 |

- **CA 最难**：POI 空间最大（9,670），地理分布覆盖整个加州，出行模式分散
- **TKY 最易**：东京签到密度高，用户行为规律（通勤主导），ZeroShot 凭语义就能做好
- **NYC 居中**：曼哈顿密度高但 POI 同质化严重

### 3. 无效输出分析

CA 有 10 个样本、TKY 有 9 个样本因模型在 10 次调用内全部输出无效 ID（不在候选池中）而未命中。因为模型未微调，缺少候选池约束。

### 4. 局限性

- 无图嵌入协同信号、无序列建模、无 CoT 推理链
- API Chat Template 依赖 OpenRouter 转换
- 候选池随机种子不可复现（与 trajectory_id 绑定）

---

## 环境与复现

```bash
cd ZeroShot_Qwen
export OPENROUTER_API_KEY="sk-or-v1-..."

uv run python main.py -d nyc   # NYC 412 样本, ~20 min
uv run python main.py -d ca    # CA  711 样本, ~50 min
uv run python main.py -d tky   # TKY 1890 样本, ~96 min
```

依赖：`openai`, `geohash2`, `tqdm`

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `main.py` | ZeroShot 推理主程序 |
| `prepare_data.py` | 数据准备脚本 |
| `run.sh` | 一键运行 |
| `README.md` | 使用说明 |
| `REPORT.md` | 本文档 |
| `output/*_results.json` | 详细评估结果 |
