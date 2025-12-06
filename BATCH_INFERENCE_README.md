# Fish Speech 批量推理脚本使用说明

## 功能说明

这个脚本提供两种模式，支持完整的批量推理工作流：

### 模式 1: 批量生成 VQ Codes (.npy 文件)
- **一次性处理多个参考音频**：批量提取参考音频的 VQ codes
- **自动命名**：根据音频文件名自动生成对应的 .npy 文件
- **完整记录**：保存每个参考音频和对应文本的映射关系

### 模式 2: 批量生成语音
- **基于参考音频高效生成**：使用提取好的 VQ codes 批量生成语音
- **模型只加载一次**：避免重复加载模型，大幅提升效率
- **批量生成**：适合生成多段文本、长文本分段处理

### 共同特点
- **自动设备检测**：支持 CUDA、MPS（Apple Silicon）、CPU
- **完善的错误处理**：单个任务失败不影响其他任务
- **显存优化**：每次生成后自动清理显存

## 使用流程

### 方式一：使用本脚本批量生成 .npy 文件（推荐）

#### 步骤 1: 配置参考音频和文本

编辑 `batch_inference.py`，配置参考音频列表和对应的文本：

```python
# 参考文本列表（每个文本对应一个参考音频）
REF_TEXT_DEFAULT_LIST = [
    "CosyVoice迎来全面升级，提供更准、更稳、更快、更好的语音生成能力。",
    "我是你大哥 是个小包工头 二哥顺路给你买回来",
    "对啊 除了你大哥呢 你还有4个哥哥"
]

# 参考音频路径列表（需要与文本列表一一对应）
REF_AUDIO_PATH_LIST = [
    "data/emb_yiya.wav",
    "data/speaker_0_reference.wav",
    "data/speaker_1_reference.wav"
]
```

#### 步骤 2: 设置模式为生成 .npy

```python
MODE = "generate_npy"  # 批量生成 .npy 文件
```

#### 步骤 3: 运行脚本

```bash
conda activate fish-audio
python batch_inference.py
```

执行后会在 `data` 目录生成：
- `data/emb_yiya_codes.npy`
- `data/speaker_0_reference_codes.npy`
- `data/speaker_1_reference_codes.npy`

### 方式二：手动生成单个 .npy 文件

如果只需要处理单个参考音频：

```bash
# 激活 conda 环境
conda activate fish-audio

# 提取参考音频的 VQ codes
python fish_speech/models/dac/inference.py \
  -i data/emb_yiya.wav \
  --checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --device cuda \
  -o data/fake.wav
```

执行后会生成 `data/fake.npy` 文件。

### 步骤 4: 批量生成语音

#### 4.1 修改配置

编辑 `batch_inference.py` 中的配置区域：

```python
# ===== 模式选择 =====
MODE = "generate_audio"  # 切换到批量生成语音模式

# 参考音频对应的文本内容（非常重要！）
# 这个文本应该是参考音频实际说的内容
REF_TEXT_DEFAULT = "CosyVoice迎来全面升级，提供更准、更稳、更快、更好的语音生成能力。"

# Prompt tokens 文件路径（之前生成的 .npy 文件）
PROMPT_TOKENS_PATH = "data/emb_yiya_codes.npy"  # 使用批量生成的文件

# 要批量生成的文本列表
TEXT_LIST = [
    "Hello, this is the first sentence.",
    "As the vocoder model has been changed, you need more room than before.",
    "Batch inference is very efficient and suitable for long text generation."
]

# 生成参数（可根据需要调整）
MAX_NEW_TOKENS = 1024      # 最大生成 token 数，文本越长需要越多
TOP_P = 0.7                # 采样参数
TEMPERATURE = 0.7          # 温度参数
REPETITION_PENALTY = 1.2   # 重复惩罚
```

#### 4.2 运行脚本

```bash
# 激活 conda 环境
conda activate fish-audio

# 运行批量推理
python batch_inference.py
```

## 输出说明

脚本会在 `data` 目录下生成音频文件：

- `data/batch_output_0.wav` - 第一个文本的生成结果
- `data/batch_output_1.wav` - 第二个文本的生成结果
- `data/batch_output_2.wav` - 第三个文本的生成结果
- ...

## 参数调优建议

### MAX_NEW_TOKENS

- 短句（< 20 字）：512
- 中等长度（20-50 字）：1024
- 长句（> 50 字）：2048

### TEMPERATURE

- 较低（0.5-0.7）：更稳定，更接近训练数据
- 中等（0.7-0.8）：平衡稳定性和多样性
- 较高（0.8-1.0）：更多样化，可能不太稳定

### REPETITION_PENALTY

- 1.0：无惩罚
- 1.1-1.3：轻微惩罚，适合大多数情况
- 1.3-1.5：较强惩罚，避免重复

## 快速开始示例

### 完整工作流示例

假设你有 3 个不同说话人的参考音频，想为每个说话人生成不同的文本：

1. **准备参考音频和文本**
   ```python
   REF_AUDIO_PATH_LIST = [
       "data/speaker_A.wav",
       "data/speaker_B.wav",
       "data/speaker_C.wav"
   ]

   REF_TEXT_DEFAULT_LIST = [
       "这是说话人A的参考文本。",
       "这是说话人B的参考文本。",
       "这是说话人C的参考文本。"
   ]
   ```

2. **批量生成 .npy 文件**
   ```python
   MODE = "generate_npy"
   ```
   运行后得到：
   - `data/speaker_A_codes.npy`
   - `data/speaker_B_codes.npy`
   - `data/speaker_C_codes.npy`

3. **使用不同说话人生成语音**
   ```python
   MODE = "generate_audio"
   PROMPT_TOKENS_PATH = "data/speaker_A_codes.npy"  # 选择说话人A
   REF_TEXT_DEFAULT = "这是说话人A的参考文本。"
   TEXT_LIST = ["要生成的文本1", "要生成的文本2"]
   ```

## 常见问题

### Q: REF_AUDIO_PATH_LIST 和 REF_TEXT_DEFAULT_LIST 长度不一致

A: 这两个列表必须一一对应，每个参考音频都需要有对应的参考文本。检查列表长度是否相同。

### Q: 提示 "参考音频文件不存在"

A: 检查 REF_AUDIO_PATH_LIST 中的路径是否正确，文件是否存在。

### Q: 提示 "Prompt tokens 文件不存在"

A: 需要先运行 `MODE = "generate_npy"` 模式，生成参考音频的 VQ codes。

### Q: 生成的音频质量不好

A: 检查以下几点：
1. `REF_TEXT_DEFAULT` 是否与参考音频的实际内容一致
2. 参考音频质量是否足够好（建议 MOS > 3.5）
3. 尝试调整 TEMPERATURE 和 REPETITION_PENALTY 参数

### Q: CUDA out of memory

A: 尝试以下方法：
1. 减少 MAX_NEW_TOKENS
2. 减少 TEXT_LIST 中的文本数量，分批处理
3. 使用更小的模型或降低精度

### Q: 生成速度慢

A: 建议：
1. 使用 GPU（CUDA）而不是 CPU
2. 如果是 Linux + CUDA，可以设置 `compile=True` 启用编译加速
3. 确保使用了 bfloat16 精度（CUDA）

## 性能参考

在不同设备上的大致生成速度（仅供参考）：

- **NVIDIA RTX 4090**：~50-100 tokens/sec
- **NVIDIA RTX 3090**：~30-60 tokens/sec
- **Apple M2 Max (MPS)**：~10-20 tokens/sec
- **CPU (i9-12900K)**：~2-5 tokens/sec

## 技术细节

### 模式 1: 批量生成 .npy 文件

工作流程：
1. **检查文件**：验证模型和参考音频文件是否存在
2. **加载 DAC 模型**：用于音频编码
3. **批量处理参考音频**：
   - 对每个参考音频：
     - 加载音频文件
     - 转换为单声道（如需要）
     - 重采样到模型采样率
     - 编码为 VQ codes
     - 保存为 .npy 文件
     - 清理显存
4. **输出文件列表**：显示所有生成的文件及其对应关系

生成的 .npy 文件命名规则：`{原音频文件名}_codes.npy`

### 模式 2: 批量生成语音

工作流程：
1. **检查文件**：验证模型、prompt tokens 等文件是否存在
2. **加载模型**：加载 Text2Semantic 模型和 DAC Codec 模型（只加载一次）
3. **加载 Prompt Tokens**：读取参考音频的 VQ codes
4. **批量生成**：
   - 对每个文本：
     - Text → Semantic Tokens（使用 Text2Semantic 模型）
     - Semantic Tokens → Audio（使用 DAC Codec 模型）
     - 保存音频文件
     - 清理显存
5. **完成**

## 配置参数完整说明

### 模式控制

```python
MODE = "generate_npy"    # 批量生成 .npy 文件
MODE = "generate_audio"  # 批量生成语音
```

### 批量生成 .npy 文件相关

```python
REF_AUDIO_PATH_LIST = [...]      # 参考音频路径列表
REF_TEXT_DEFAULT_LIST = [...]    # 参考文本列表（与音频一一对应）
```

### 批量生成语音相关

```python
PROMPT_TOKENS_PATH = "..."       # 使用哪个 .npy 文件
REF_TEXT_DEFAULT = "..."         # 该 .npy 对应的参考文本
TEXT_LIST = [...]                # 要生成的文本列表
```

## 许可证

本脚本基于 Fish Speech 项目，遵循其开源协议。
