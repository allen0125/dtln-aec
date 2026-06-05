# DTLN-AEC：声学回声消除（Acoustic Echo Cancellation）复现

基于 DTLN 框架的双信号声学回声消除模型复现。模型接收 **带回声的麦克风信号** 与 **远端/loopback 参考信号**，输出去除回声后的 **近端语音**。

- 论文：[Acoustic Echo Cancellation with the Dual-Signal Transformation LSTM Network](https://arxiv.org/pdf/2010.14337.pdf)
- 参考实现（推理）：[breizhn/DTLN-aec](https://github.com/breizhn/DTLN-aec)
- 基础框架（降噪版 DTLN）：[breizhn/DTLN](https://github.com/breizhn/DTLN)

本仓库的 DTLN-AEC 模型参考上述论文与 DTLN 源码复现，并针对工作中的数据搭建了完整的数据处理与训练流程。

> 说明：本项目是工作中需要复现的论文之一，训练所用为公司内部数据（约 50G），因数据涉密不随仓库提供；下文脚本中的绝对路径需替换为你自己的数据目录。

---

## 模型架构

模型由两个串联的「分离核心（separation core）」组成，每个核心是 2 层 LSTM + 全连接 + sigmoid，用于预测一个 mask。

**第一核心（时频域）**

1. 对 mic 与 lpb 分别做 STFT（帧长 512 / 32ms，帧移 128 / 8ms，rFFT → 257 个频点）。
2. 对两路 **归一化的 log 功率谱**（各自独立做 instant layer normalization，`norm_stft=True`）在 **频率/特征维（`axis=-1`）** 拼接，送入第一核心。
3. 预测的 mask 与 **未归一化的 mic 幅度谱** 相乘，再用 **mic 的相位** 经 iFFT 还原为时域帧。

**第二核心（学习到的特征域）**

1. 用共享权重的 1D-Conv 编码器（`encoder_size=512`）分别编码「第一核心输出的时域帧」与「lpb 的分帧信号」。
2. 两路编码各自做 iLN 后拼接，送入第二核心。
3. 预测的 mask 与 **未归一化的第一核心输出特征** 相乘，再经 1D-Conv 解码器 + overlap-add 还原波形，即估计的近端语音。

**损失函数**：时域 SNR loss。**优化器**：Adam（`lr=1e-3`，`clipnorm=3`），配合 `ReduceLROnPlateau` 与 `EarlyStopping`。

### 主要超参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `fs` | 16000 | 采样率 |
| `blockLen` | 512 | 帧长（32 ms）|
| `block_shift` | 128 | 帧移（8 ms）|
| FFT / 频点数 | 512 / 257 | rFFT，`blockLen//2 + 1` |
| `numUnits` | 512 | 每层 LSTM 单元数 |
| `numLayer` | 2 | 每个核心的 LSTM 层数 |
| `encoder_size` | 512 | 1D-Conv 特征维度（对齐论文）|
| `dropout` | 0.25 | LSTM 层间 dropout |
| `batchsize` | 32 | |
| `len_samples` | 15 | 训练片段时长（秒）|
| `max_epochs` | 200 | 早停 patience=10 |

---

## 目录结构

```
dtln-aec/
├── aec/
│   ├── model/
│   │   ├── dtln_aec_model.py   # 模型定义、编译与训练逻辑
│   │   ├── aec_blood.py        # tf.data 数据生成器
│   │   └── run_training.py     # 训练入口
│   └── scripts/
│       ├── fetch_lpb_audio.py  # 下载 loopback(远端)音轨
│       ├── align_wav.py        # 对齐 mic/lpb 并切分为定长片段
│       └── overlay_wav.py      # 合成回声并生成 mixed 训练数据
├── data/
│   ├── origin_data/            # 原始数据与 ffmpeg 转换脚本
│   └── aec_sample/             # 样例音频(mic/lpb/output)
├── requirements.txt
└── setup.py
```

---

## 环境依赖

- Python ≥ 3.6，TensorFlow 2.x（建议 GPU）
- 主要依赖：`tensorflow`、`numpy`、`scipy`、`soundfile`、`librosa`、`wavinfo`、`requests`
- 数据预处理需要 `ffmpeg`

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

---

## 数据准备

整体流程：原始音频 → 转 16kHz 单声道 wav → 获取 loopback → 对齐切分 → 合成回声生成 mixed → 训练。

三类音频的语义：

| 文件 | 含义 | 在训练中的角色 |
| --- | --- | --- |
| `*-mic.wav` | 近端麦克风信号 | 训练 **目标**（干净近端 `s(n)`）|
| `*-lpb.wav` | loopback / 远端参考信号 | 模型 **参考输入** |
| `*-mix-echo.wav` | 由近端 + 合成回声构成的带回声信号 | 模型 **主输入** |

> 关键：模型输入是 `(带回声 mic, 远端 lpb)`，目标是 `干净近端 mic`。数据生成器据此 yield `((mixed, lpb), mic)`。

**1. 格式转换** —— 将 mic 源文件统一转为 16kHz 单声道 wav，命名为 `*-mic.wav`：

```bash
cd data/origin_data && bash ffmpeg_foo_to_wav.sh
```

**2. 获取 loopback** —— `fetch_lpb_audio.py` 依据 CSV 中的映射下载远端原始音轨，再用 ffmpeg 转为 `*-lpb.wav`：

```bash
python aec/scripts/fetch_lpb_audio.py
```

**3. 对齐与切分** —— `align_wav.py` 将 mic/lpb 截到等长、跨文件拼接残余尾帧，并切为 30s 片段，输出命名为 `realname-aligned-{i}-mic.wav` / `-lpb.wav`：

```bash
python aec/scripts/align_wav.py
```

**4. 合成回声、生成 mixed** —— `overlay_wav.py` 从 lpb 合成真实回声（RIR 卷积 → 混响尾随机衰减 `[-25,0]dB` → 随机延迟 → 带通 100–400/6000–7500 Hz → 按随机 SER 与近端混合），输出 `*-mix-echo.wav`：

```bash
python aec/scripts/overlay_wav.py
```

> 注意：用于训练的 `*-lpb.wav` 始终保持 **原始远端参考**，回声合成只写入 mixed —— 让网络自行估计回声路径，避免学成「逐样本相减」的退化解。如有实测房间冲激响应，可在 `overlay_wav.py` 中设置 `rir_dir` 加载，否则会自动生成指数衰减的随机 RIR。

各脚本顶部的目录路径需按你的环境修改。

---

## 训练

在 `aec/model/run_training.py` 中配置训练/验证集的 `mixed`、`mic`、`lpb` 三个目录，然后：

```bash
python aec/model/run_training.py
```

训练过程会输出到 `./models_<run_name>/`，包含最优权重 `<run_name>.h5`、`training_<run_name>.log` 以及基于 `val_loss` 的学习率衰减与早停。

---

## 推理 / 验证

可参考上游 [breizhn/DTLN-aec](https://github.com/breizhn/DTLN-aec) 的 `run_aec.py`：将训练得到的权重转换/加载后，对 `(mic, lpb)` 逐块推理，输出去回声的近端语音。`data/aec_sample/` 下提供了 mic/lpb 及一份 output 样例可供对照。

---

## 结果

- 数据集：约 50G 公司内部 AEC 数据（不随仓库提供）。
- 训练时长：约 50 小时（早停收敛）。
- 效果：回声消除效果良好，达到了可用于业务的水平；当时的具体客观指标（如 ERLE / PESQ）未在此仓库记录。

---

## 复现要点（踩坑记录）

1. **拼接维度**：mic 与 lpb 应在 **频率/特征维** 拼接。STFT 后张量形状为 `(batch, frames, freq)`，因此 `tf.concat` 的 `axis` 应为 `-1`（最后一维），**不是** `0`（batch 维）。
2. **第二核心输入**：是分帧后的信号——第一核心输出经 iFFT 得到的时域帧，以及 lpb 经 `tf.signal.frame` 得到的帧。
3. **数据映射**：模型输入 `[带回声 mic, 远端 lpb]`，目标 `干净近端 mic`，即 generator 须 yield `((mixed, lpb), mic)`；切勿把 lpb 当目标、mic 当参考。
4. **归一化**：第一核心按论文使用 **归一化的 log 功率谱**（`norm_stft=True`）；`encoder_size` 与论文一致取 `512`。
5. **回声合成**：不要直接 overlay 原始 lpb，而要对 lpb 做 RIR 卷积 + 随机增益 + 延迟 + 带通（见 `scripts/overlay_wav.py`）；训练用的 lpb 文件保持原始远端参考。
