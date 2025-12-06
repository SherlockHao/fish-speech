#!/usr/bin/env python3
"""
Fish Speech 批量推理脚本
功能：
1. 批量生成参考音频的 VQ codes (.npy 文件)
2. 基于参考音频批量生成多个文本的语音
"""
import os
import sys
import torch
import soundfile as sf
import numpy as np
import torchaudio
from loguru import logger

# 导入 Fish Speech 的推理模块
from fish_speech.models.text2semantic.inference import init_model, generate_long
from fish_speech.models.dac.inference import load_model as load_dac_model

# ============================= 配置区域 =============================

# ===== 模式选择 =====
# "generate_npy" - 批量生成参考音频的 .npy 文件
# "generate_audio" - 批量生成文本的语音
MODE = "generate_audio"  # 修改这里来切换模式

# 参考音频对应的文本内容 (非常重要，用于提取音色和韵律)
REF_TEXT_DEFAULT = "CosyVoice迎来全面升级，提供更准、更稳、更快、更好的语音生成能力。"

# 模型路径
CHECKPOINT_PATH = "checkpoints/openaudio-s1-mini"

# Prompt tokens 文件路径（之前用 DAC encode 提取好的参考音频的 VQ codes）
PROMPT_TOKENS_PATH = "data/speaker_0_reference_codes.npy"

REF_TEXT_DEFAULT_LIST = [
    "CosyVoice迎来全面升级，提供更准、更稳、更快、更好的语音生成能力。",
    "我是你大哥 是个小包工头 二哥顺路给你买回来 以后谁欺负你 今天都4个菜了 平时咱们就一个菜 果然是个吃不了苦的孩子",
    "对啊 除了你大哥呢 你还有4个哥哥 总共5个哥哥 吃饭去 欣欣 你是不是 不喜欢这些菜啊 吃吧"
]

REF_AUDIO_PATH_LIST = [
    "data/emb_yiya.wav",
    "data/speaker_0_reference.wav",
    "data/speaker_1_reference.wav"
]

# 输出目录
OUTPUT_DIR = "data"

# 这里的 text_list 是你想批量生成的文本列表
TEXT_LIST = [
    "Hello, this is the first sentence.",
    "As the vocoder model has been changed, you need more room than before.",
    "Batch inference is very efficient and suitable for long text generation."
]

# 生成参数
MAX_NEW_TOKENS = 1024  # 根据文本长度调整，文本越长需要越多tokens
TOP_P = 0.7
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.2

# ======================================================================

def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def check_files(mode="generate_audio"):
    """检查必要文件是否存在"""
    errors = []

    # 检查模型目录
    if not os.path.exists(CHECKPOINT_PATH):
        errors.append(f"模型目录不存在: {CHECKPOINT_PATH}")

    if mode == "generate_audio":
        # 检查 prompt tokens 文件
        if not os.path.exists(PROMPT_TOKENS_PATH):
            errors.append(f"Prompt tokens 文件不存在: {PROMPT_TOKENS_PATH}")
            errors.append("请先运行模式 MODE='generate_npy' 来生成 .npy 文件")
    elif mode == "generate_npy":
        # 检查参考音频文件
        for ref_audio in REF_AUDIO_PATH_LIST:
            if not os.path.exists(ref_audio):
                errors.append(f"参考音频文件不存在: {ref_audio}")

    # 检查输出目录
    if not os.path.exists(OUTPUT_DIR):
        logger.info(f"创建输出目录: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)

def generate_prompt_tokens_batch(dac_model, device):
    """批量生成参考音频的 VQ codes (.npy 文件)"""
    logger.info(f"\n开始批量生成 {len(REF_AUDIO_PATH_LIST)} 个参考音频的 VQ codes...")

    if len(REF_AUDIO_PATH_LIST) != len(REF_TEXT_DEFAULT_LIST):
        logger.error(f"❌ REF_AUDIO_PATH_LIST 和 REF_TEXT_DEFAULT_LIST 长度不匹配")
        logger.error(f"   音频数量: {len(REF_AUDIO_PATH_LIST)}, 文本数量: {len(REF_TEXT_DEFAULT_LIST)}")
        sys.exit(1)

    generated_files = []

    for i, (ref_audio, ref_text) in enumerate(zip(REF_AUDIO_PATH_LIST, REF_TEXT_DEFAULT_LIST)):
        logger.info(f"\n{'='*60}")
        logger.info(f"正在处理第 {i+1}/{len(REF_AUDIO_PATH_LIST)} 个参考音频")
        logger.info(f"  音频: {ref_audio}")
        logger.info(f"  文本: {ref_text}")

        try:
            # 1. 加载音频
            audio, sr = torchaudio.load(str(ref_audio))

            # 转换为单声道
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
                logger.info(f"  已转换为单声道")

            # 重采样到模型采样率
            audio = torchaudio.functional.resample(audio, sr, dac_model.sample_rate)
            logger.info(f"  音频时长: {audio.shape[1] / dac_model.sample_rate:.2f} 秒")

            # 2. 编码为 VQ codes
            audios = audio[None].to(device)
            audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)

            with torch.no_grad():
                indices, _ = dac_model.encode(audios, audio_lengths)

            if indices.ndim == 3:
                indices = indices[0]

            logger.info(f"  生成的 VQ codes shape: {indices.shape}")

            # 3. 保存 .npy 文件
            # 根据参考音频文件名生成对应的 .npy 文件名
            audio_basename = os.path.splitext(os.path.basename(ref_audio))[0]
            npy_filename = os.path.join(OUTPUT_DIR, f"{audio_basename}_codes.npy")

            np.save(npy_filename, indices.cpu().numpy())
            logger.info(f"✅ 已保存: {npy_filename}")

            generated_files.append({
                'npy_file': npy_filename,
                'ref_audio': ref_audio,
                'ref_text': ref_text,
                'index': i
            })

            # 清理显存
            del audio, audios, indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ 处理第 {i+1} 个参考音频失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 批量生成 VQ codes 完成！共生成 {len(generated_files)} 个文件")
    logger.info(f"\n生成的文件列表：")
    for item in generated_files:
        logger.info(f"  [{item['index']}] {item['npy_file']}")
        logger.info(f"      参考音频: {item['ref_audio']}")
        logger.info(f"      参考文本: {item['ref_text']}")

    return generated_files

def main():
    logger.info("🐟 Fish Speech 批量推理脚本启动...")
    logger.info(f"当前模式: {MODE}")

    # 1. 检查文件
    check_files(mode=MODE)

    # 2. 检测设备
    device = get_device()
    logger.info(f"使用设备: {device}")

    # 3. 设置精度
    if device == "cuda":
        precision = torch.bfloat16
    else:
        precision = torch.float32

    # 4. 加载 DAC 模型（两种模式都需要）
    logger.info("正在加载 DAC (Codec) 模型...")
    try:
        dac_model = load_dac_model(
            config_name="modded_dac_vq",
            checkpoint_path=os.path.join(CHECKPOINT_PATH, "codec.pth"),
            device=device
        )
        logger.info("✅ DAC 模型加载完成")
    except Exception as e:
        logger.error(f"❌ 加载 DAC 模型失败: {e}")
        sys.exit(1)

    # ===== 模式 1: 批量生成 .npy 文件 =====
    if MODE == "generate_npy":
        generate_prompt_tokens_batch(dac_model, device)
        return

    # ===== 模式 2: 批量生成语音 =====
    # 5. 加载 Text2Semantic 模型（仅生成语音时需要）
    logger.info("正在加载 Text2Semantic 模型...")
    try:
        llama_model, decode_one_token = init_model(
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            precision=precision,
            compile=False  # 如果是 Linux + CUDA 可以设为 True 加速
        )
        logger.info("✅ Text2Semantic 模型加载完成")
    except Exception as e:
        logger.error(f"❌ 加载 Text2Semantic 模型失败: {e}")
        sys.exit(1)

    # 5. 加载 Prompt Tokens (如果多个任务共用同一个 prompt，只需加载一次)
    logger.info(f"加载 Prompt Tokens: {PROMPT_TOKENS_PATH}")
    try:
        prompt_tokens = np.load(PROMPT_TOKENS_PATH)
        prompt_tokens = torch.from_numpy(prompt_tokens).to(device).long()
        if prompt_tokens.ndim == 3:
            prompt_tokens = prompt_tokens[0]  # 移除batch维度
        logger.info(f"✅ Prompt Tokens 加载完成，shape: {prompt_tokens.shape}")
    except Exception as e:
        logger.error(f"❌ 加载 Prompt Tokens 失败: {e}")
        sys.exit(1)

    # 6. 批量循环生成
    logger.info(f"\n开始批量生成 {len(TEXT_LIST)} 个文本...")

    for i, text in enumerate(TEXT_LIST):
        logger.info(f"\n{'='*60}")
        logger.info(f"正在生成第 {i+1}/{len(TEXT_LIST)} 条: {text}")

        try:
            # 步骤 A: 文本转语义 Token (Text to Semantic)
            # 使用 generate_long 生成器函数
            codes = None
            for response in generate_long(
                model=llama_model,
                device=device,
                decode_one_token=decode_one_token,
                text=text,
                prompt_text=REF_TEXT_DEFAULT,
                prompt_tokens=prompt_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                num_samples=1
            ):
                if response.action == "sample":
                    codes = response.codes
                    logger.info(f"✅ 生成语义 tokens，shape: {codes.shape}")

            if codes is None:
                logger.error(f"❌ 未能生成语义 tokens")
                continue

            # 步骤 B: 语义 Token 转语音 (Semantic to Waveform)
            # 使用 DAC 模型解码
            if codes.ndim == 2:
                codes = codes.unsqueeze(0)  # 添加batch维度 [1, codebook, T]

            codes_lens = torch.tensor([codes.shape[-1]], device=device, dtype=torch.long)

            with torch.no_grad():
                fake_audios, _ = dac_model.decode(codes, codes_lens)

            # 保存音频
            output_filename = os.path.join(OUTPUT_DIR, f"batch_output_{i}.wav")
            fake_audio = fake_audios[0, 0].float().cpu().numpy()
            sf.write(output_filename, fake_audio, dac_model.sample_rate)

            logger.info(f"✅ 已保存: {output_filename}")
            logger.info(f"   时长: {len(fake_audio) / dac_model.sample_rate:.2f} 秒")

            # 清理显存
            del codes, fake_audios, fake_audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ 生成第 {i+1} 条失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\n{'='*60}")
    logger.info("🎉 批量任务完成！")

if __name__ == "__main__":
    main()