# demucs-wgpu 无时长、无质量损失：根治设计

**日期**：2026-10-10（CST）  
**范围**：demucs-wgpu `pipeline.rs` 全缓冲 OLA → 无界长度流式 OLA；对齐当前 OneAsr / demucs-core 默认行为。  
**非目标**：把「只要 vocals」当成产品目标。vocals-only 可作内存优化，但根治必须让**同一套 `SeparateOptions` 下的完整 OLA 数学**在任意时长上等价于今日全缓冲实现。

---

## 1. 质量基线跑数（本机）

### 1.1 输入

| 项 | 值 |
|----|-----|
| 路径 | `/workspace/oneasr-baseline/media/user_test/demucs_baseline.wav` |
| 来源 | 用户附件复制（hash 文件名 `4cf47995…c724.wav`） |
| 格式 | PCM s16le，**44100 Hz，单声道**，时长 **176.309 s**（7 775 232 frames） |
| 体积 | ~14.8 MB |

 demucs / OneAsr 读入时会把 mono **复制为立体声**（L=R），再按模型 44.1 kHz 分离。

### 1.2 命令与实现钉扎

- **实现**：`demucs-wgpu` @ `ca8359aca0b890fcf71dd1a475f8c2675261a51b`（与 OneAsr `Cargo.toml` 中 `demucs-core` rev 一致）
- **权重**：`/workspace/oneasr-runtime/models/htdemucs_ft/htdemucs_ft_vocals.safetensors`  
  （sha256 `68854b0d7c2b3274723b5761f6fd9f5a…`，与 OneAsr catalog 一致；按 `ModelVariant::FourStem` + `--stem vocals` 加载，与 OneAsr `DemucsSeparatorAdapter` 相同）
- **设备**：本机 `demucs adapters` → *no wgpu adapters* → **host CPU**（日志：`using the host path`）
- **CLI**（等价 OneAsr `separate_with_progress` → `SeparateOptions::default()`）：

```bash
demucs separate demucs_baseline.wav \
  -o demucs_out/vocals_shifts1.wav \
  --model …/htdemucs_ft_vocals.safetensors \
  --device auto --stem vocals \
  --shifts 1 --overlap 0.25 --timings
```

### 1.3 生效设置（必须写入对照）

来自 `SeparateOptions::default()` + 模型 config `segment = 39/5`：

| 参数 | 值 | 出处 |
|------|-----|------|
| `shifts` | **1** | `pipeline.rs` Default；OneAsr 未覆盖 |
| `overlap` | **0.25** | 同上 |
| `transition_power` | **1.0** | 同上 |
| `segment` | **7.8 s**（`None` → `model.config.segment`） | `config.rs` `39.0/5.0` |
| `segment_length` | **343 980** samples | `(samplerate as f64 * segment_seconds) as usize` |
| `stride` | **257 985** | `((1.0 - overlap) * segment_length as f64) as usize` |
| 重叠区 | **85 995** samples（≈1.95 s） | `segment_length - stride` |
| chunk 数 | **30** | `num_chunks_for`；日志 `30 chunks` |
| `max_shift` | **22 050**（0.5 s） | `shifts>0` 分支 |
| 本 port 首个 shift offset | **11 660**（`ShiftRng::new(0x5eed_1234)`，**可复现**） | 非 Python 无种子 `random` |

### 1.4 输出产物

| 项 | 值 |
|----|-----|
| 主基线 WAV | `/workspace/oneasr-baseline/media/user_test/demucs_out/vocals.wav` |
| 同内容备份 | `…/vocals_shifts1.wav` |
| 格式 | PCM **f32le**，44100 Hz，**立体声**，时长与输入一致 176.309 s |
| 体积 | ~62.2 MB |
| 峰值 | 日志 `peak 0.9176` |
| 耗时 | 分离 **63.74 s**，RTFx ≈ **2.77×**（CPU）；含加载 end-to-end ≈ 2.74× |

后续任何「流式 / 无界」改动必须能对该文件做 SNR / 逐样本 diff：目标是 **bit-exact 或数值上不可闻的 near-exact**（见 §2）。

---

## 2. 真流式 overlap-add 能否对齐当前全缓冲输出？

结论先说：

- **OLA 本体（`split_pass` + 三角窗加权累加 / `sum_weight` 归一）**：在相同 `segment/overlap/transition_power`、相同 forward、相同浮点累加顺序下，用**环形缓冲 + 冲刷已封闭区间**可以做到与今日全缓冲 **bit-exact**。
- **整轨全局归一化 `normalize` / `denormalize`**：单遍在线流式**做不到**与今日 bit-exact，除非改成**两遍**（先扫 mean/std，再流式推理）或接受局部归一（**有质量差**）。
- **`shifts ≥ 1`**：数学上仍可 bit-exact，但实现要从「整轨 pad → 多遍全长 `split_pass` → 平均」改成带 `max_shift` 延迟的多路 OLA 或落盘再平均；naive 切段拼接 **不能**对齐。
- **外层乱切窗、各自 normalize、硬拼接**：必然有边界伪影，**不能**当根治。

### 2.1 当前全缓冲管道（引用 `pipeline.rs`）

1. **全局归一**（`api.rs` → `normalize`）  
   对整轨 `(batch, channels, T)` 做通道平均得到 `reference[t]`，再  
   `mean = sum(ref)/(B·T)`，`std = sqrt(var) + 1e-8`（样本方差分母 `count-1`），  
   `wav' = (wav - mean) / std`。  
   输出侧 `denormalize`：`y * std + mean`。  
   → **统计量依赖全长 T**。流式若用滑动/分块 mean，结果必变。

2. **shifts 循环**（`separate_forwarding`，`shifts > 0`）  
   - `padded_chunk(mix, 0, length, length + 2*max_shift)` 左右各垫 `max_shift`；  
   - 每遍 `offset = rng.randint(max_shift)`，取切片长度 `length + max_shift - offset`；  
   - 对该切片跑完整 `split_pass`；  
   - trim 回原长度后累加，最后 `× 1/shifts`。  
   本 port 的 `ShiftRng` **有固定种子**，故 offset **可预知**（基线首遍 11660），便于对照测试。

3. **`split_pass`（`split=True` 分支）**  
   - `stride = ((1-overlap)*segment_length) as usize`；  
   - `weight = transition_weight(segment_length, transition_power)`（三角窗再 `powf`）；  
   - 分配 **`out: Array4(B, sources, C, length)`** 与 **`sum_weight: Vec<f32>(length)`** ← **O(T) 内存根因**；  
   - 对每个起点 `offset = 0, stride, 2·stride, …`：  
     `padded_chunk` → model forward → `center_trim` →  
     `out[…, start+t] += weight[t] * trimmed[t]`，`sum_weight[start+t] += weight[t]`；  
   - 最后对所有 `t`：`out[…] /= sum_weight[t]`。

4. **何时某个样本的 OLA 「封闭」**  
   段起点为 `S` 的 chunk 只写入区间 `[S, S+chunk_len)`。  
   下一个段起点为 `S+stride`，写入从 `S+stride` 开始。  
   因此：**在完成起点为 `S` 的段之后，区间 `[0, S+stride)` 上不会再有未来段写入**；该前缀的 `sum_weight` 与加权和已最终确定，可立即 `/= sum_weight`（再乘全局 `std` 加 `mean`）并冲刷出环。  
   环上只需保留约 **`segment_length`（更紧：`segment_length - stride` 的开重叠窗 + 当前段）** 量级的活动缓冲，与总时长无关。

5. **浮点顺序**  
   代码注释写明：分组 batch（`DEMUCS_BATCH`）只改调度，**按段顺序 splice** 时与逐段相同（默认 group=1 时累加序列字面相同）。流式冲刷必须保持：**段顺序不变、段内 `b/src/c/t` 累加顺序不变**，否则仅舍入级差异也可能出现。GPU vs CPU 路径本身就不是 bit-exact；对照时应固定同一 backend。

### 2.2 「真流式 ring / flush」vs 全缓冲：对齐等级

| 条件 | 与当前全缓冲关系 |
|------|------------------|
| 两遍全局 mean/std + 同 options + 同 RNG shift + 同 backend + 按上式 flush | **可 bit-exact**（OLA+归一） |
| 单遍、用局部/在线归一代替全局 `normalize` | **近、但非 exact**；长轨响度漂移处可闻 |
| 保留全局归一，但独立切段、段边界硬切或错误提前 `/= sum_weight` | **边界失真**，不能当无质量损失 |
| `shifts>0` 却忽略 pad/平均，只跑一遍 | **系统性偏差**（与今日 default `shifts=1` 不一致） |

**回答任务第 2 点**：真流式 OLA（ring + flush past segments）**可以**在相同设置下匹配当前输出的 bit-exact / near-exact；前提是 **不要动全局归一与 shifts 语义**，只把 `out`/`sum_weight` 的存储从 `O(T)` 换成 `O(segment)` 的滑动窗，并严格按 `pipeline.rs` 的封闭点冲刷。做不到 bit-exact 的，是「假装流式却改归一/改切段策略」的捷径，不是 OLA 数学本身。

---

## 3. 根治设计：无界时长且不降质量

### 3.1 设计目标

1. **质量**：同一 `SeparateOptions`、同一模型与 backend，对流式结果与全缓冲结果在基线 WAV（及更长人造轨）上 bit-exact（测试门禁），或至少 SNR ≥ 当前 GPU↔host 对齐量级（>100 dB）且无边界咔哒。  
2. **内存**：峰值 ≈ **输入回看窗 + 一段模型激活 + O(segment)×sources×channels 的 OLA 环 + 权重**，**不随 T 线性涨**（今日 2h 十数 GB 的 Array4 必须消失）。  
3. **语义**：默认仍 `shifts=1, overlap=0.25, segment=7.8s`；不是「为 ASR 砍成 vocals-only 特例」才省内存。  
4. **API**：库内提供流式入口；OneAsr 只换调用，不自己发明切段拼接。

### 3.2 推荐架构（demucs-core 内）

```
Pass A（可选与解码合并）: 流式读 PCM → 按 normalize 公式累加 Σref、Σref²、count
                           → GlobalNorm { mean, std }

Pass B: 再次流式读（或 mmap）
        for each shift pass k = 0..shifts-1:
            offset_k = ShiftRng 预生成（与现实现同种子）
            StreamingSplitPass:
              input 环形缓冲（≥ segment_length + 尾块两侧 pad 所需上下文）
              out_acc / sum_weight 环形缓冲（长度 ≥ segment_length）
              cursor S = 0, stride, …
              每完成段 S:
                flush [flush_pos, S+stride) → /sum_weight → *std+mean → 写入 Writer
              尾块 short chunk: 与 padded_chunk/center_trim 完全一致
        shifts>1: 见 §3.3
```

**核心不变量（直接对应现码）**

- `segment_length`、`stride`、`transition_weight` 计算式不变。  
- `padded_chunk` / `center_trim` 语义不变（尾块仍要左右上下文与零垫）。  
- `splice` 权重累加顺序不变。  
- flush 点：完成起点 `S` 后，冲刷到 `S+stride`（尾段冲刷到 `length`）。  
- `normalize` 的 mean/std **必须是整轨**（Pass A），不得改为 per-segment。

### 3.3 `shifts` 策略（保质量）

今日默认 `shifts=1`，必须覆盖：

| 方案 | 内存 | 磁盘 | 与现输出 |
|------|------|------|----------|
| **A. 多路锁步（推荐 shifts 小）** | O(shifts × segment) | 低 | bit-exact：每样本等所有 shift 贡献齐再写 |
| **B. 逐 shift 写临时 stem，再流式平均** | O(segment) | O(shifts × T) | bit-exact |
| **C. 强行 shifts=0** | 最省 | 低 | **有质量差**，仅作可选快档，**不能**当根治默认 |

对 OneAsr 长视频：推荐 **A（shifts=1 时就是单路 + 输入端 `max_shift` 预卷）**：  
输入逻辑时间 `t` 对应取 `padded[t + max_shift - offset]` 的流视图，首尾零垫与现 `padded_chunk` 一致；输出在 trim 规则下与现 trim 切片一致。

### 3.4 内存布局（相对今日）

| 缓冲 | 今日（全长） | 根治后 |
|------|-------------|--------|
| L/R 或 mix f32 | O(T) 常驻 | 文件/mmap + O(segment) 窗 |
| `normalize` 克隆 | O(T) | Pass A 标量统计；Pass B 现场 `(x-mean)/std` |
| `out` Array4 全 sources | O(sources·C·T) ≈ 四轨全长 | **环 O(sources·C·segment)**；若调用方只要部分 stem，环可只留请求轨（**优化**，非质量开关） |
| `sum_weight` | O(T) | O(segment) |
| 模型单段激活 | O(1) 段 | 不变 |
| 输出 WAV | 最后一次性写 | **增量写**（s16 或 f32）；避免再持有全长 `Stem` `Vec` |

2h @ 44.1 kHz 立体声：今日峰值十余 GB → 根治后主机侧音频工作集应落在 **百 MB 级（段+环+回看）**，外加模型权重与 GPU arena（本就按段）。

### 3.5 OneAsr 侧配合（非根治本体，但是落地必需）

1. 调用新 API：`separate_file_streaming` / `separate_reader`，scratch 目录直接收 `vocals.wav`（或四轨）。  
2. 去掉 `read_stereo_f32` 整轨 `Vec` 双峰；解码可用 ffmpeg 管式或分块 hound。  
3. 进度：仍按段 `done/total`；`total` 在知 T 或 Pass A 后可知（与现 `num_chunks_for` 一致）。  
4. **不要**在 OneAsr 外层「切 5 分钟块再 ffmpeg concat」——除非每块实现与上完全相同的跨块 OLA 状态机；那只是把 bug 挪到应用层。  
5. vocals-only 少物化其它 stem：降内存，**不改变** vocals lane 数值（已有 bag/shard 测试语义）；与无界是正交优化。

### 3.6 验收门禁（必须写进 CI）

1. **短轨 bit-exact**：现有测试音 / 本基线 176 s，`shifts=0` 与 `shifts=1`（固定 RNG）分别对比 `separate` vs `separate_streaming`，`max|Δ|=0`。  
2. **合成长轨**：重复基线或噪声到 ≥30–60 min（CI 可用更短段长 `segment=0.25` 测 OLA 状态机），内存上限断言（如 RSS 增量 < 固定阈值）。  
3. **边界**：人为在 stride 边界放脉冲，确认无咔哒（流式 vs 全缓冲 diff=0）。  
4. **回归质量基线**：对本文件 `vocals.wav`，流式改动后 SNR 对全缓冲参考 → ∞（bit-exact）或文档化的阈值。  
5. **禁止**用「听感差不多」代替上述门禁。

### 3.7 明确不采纳的「假根治」

- 缩小 `segment` / 降 `overlap` / 默认 `shifts=0` 当唯一修复（质量锚被挪走）。  
- 按静音/VAD 切段分别 Demucs 再拼接（边界与全局归一双杀）。  
- 只改 OneAsr 写 s16、减 shifts，却仍分配全长 `Array4`（缓解 OOM，**不**解决无界）。  
- 声称「流式 OLA 必然损失质量」——与 `pipeline.rs` 数学不符；损失来自改归一/改封闭条件，而非 ring 本身。

### 3.8 建议落地顺序

1. **demucs-core**：抽出 `GlobalNorm` 扫描 + `StreamingSplitPass`（先 `shifts=0` 门禁 bit-exact）。  
2. 接上 **shifts≥1**（锁步或临时文件平均），默认行为与 `SeparateOptions::default()` 一致。  
3. CLI：`demucs separate --streaming`（或长文件自动走流式）。  
4. OneAsr：换适配器读/写路径；Issue #17 类 2h OOM 用长轨压测关闭。  
5. 保留本目录 `demucs_baseline.wav` + `demucs_out/vocals.wav` 为质量金样。

---

## 4. 文件索引

| 路径 | 说明 |
|------|------|
| `media/user_test/demucs_baseline.wav` | 用户质量基线输入 |
| `media/user_test/demucs_out/vocals.wav` | 当前全缓冲 OLA 输出（OneAsr 默认设置） |
| `media/user_test/demucs_out/vocals_shifts1.wav` | 同上备份 |
| `notes/demucs_unlimited_quality.md` | 本文 |
| demucs-core `crates/demucs-core/src/demucs/pipeline.rs` | 归一 / shifts / `split_pass` / OLA 数学权威来源 |
| OneAsr `crates/oneasr-core/src/engine/local/demucs.rs` | 默认 `separate_with_progress` → `SeparateOptions::default()` |

---

## 5. 一句话结论

**根治 = 把 `split_pass` 的全长 `out`/`sum_weight` 换成按 `S+stride` 封闭规则冲刷的环形 OLA，并两遍保留全局 `normalize`；`shifts` 用可复现 RNG 做锁步或落盘平均。**  
这样无界时长与当前全缓冲输出可 bit-exact 对齐；任何改归一、假切段、默默改 `shifts/overlap` 的方案都不是「无质量损失」根治。
