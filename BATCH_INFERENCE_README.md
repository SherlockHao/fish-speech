# Fish Speech 批量推理脚本使用说明

## 功能说明

这个脚本可以基于同一个参考音频（.npy文件），高效地批量生成多个文本的语音。特点：

- **模型只加载一次**：避免重复加载模型，大幅提升效率
- **批量生成**：适合生成多段文本、长文本分段处理
- **自动设备检测**：支持 CUDA、MPS（Apple Silicon）、CPU
- **完善的错误处理**：每个文本生成失败不会影响其他文本
- **显存优化**：每次生成后自动清理显存

## 使用前准备

### 1. 准备参考音频的 VQ Codes（.npy 文件）

在批量生成之前，需要先提取参考音频的 VQ codes：

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

执行后会生成 `data/fake.npy` 文件，这就是参考音频的 VQ codes。

### 2. 修改配置

编辑 `batch_inference.py` 中的配置区域：

```python
# ============================= 配置区域 =============================
SPEAKER_NAME = 'yiya'  # 说话人名称（可选，仅用于标识）

# 参考音频对应的文本内容（非常重要！）
# 这个文本应该是参考音频实际说的内容
REF_TEXT_DEFAULT = "CosyVoice迎来全面升级，提供更准、更稳、更快、更好的语音生成能力。"

# 模型路径
CHECKPOINT_PATH = "checkpoints/openaudio-s1-mini"

# Prompt tokens 文件路径（上一步生成的 .npy 文件）
PROMPT_TOKENS_PATH = "data/fake.npy"

# 输出目录
OUTPUT_DIR = "data"

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
# ======================================================================
```

## 运行脚本

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

## 常见问题

### Q: 提示 "Prompt tokens 文件不存在"

A: 需要先运行步骤1，提取参考音频的 VQ codes。

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

脚本工作流程：

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

## 许可证

本脚本基于 Fish Speech 项目，遵循其开源协议。
