import os
import sys
import subprocess
from pathlib import Path

# ================= 配置区域 (Configuration) =================
# 1. 想要生成的文本 (Target Text)
TARGET_TEXT = "你好，这是一个由Fish Speech自动生成的语音克隆测试。"

SPEAKER_NAME = 'yiya'

# 2. 参考音频路径 (必须是 .wav 格式，放在根目录)
REF_AUDIO_PATH = "data/emb_" + SPEAKER_NAME + ".wav"

# 3. 参考音频对应的文本内容 (非常重要，用于提取音色和韵律)
# 如果你根目录下有 ref.txt，脚本会优先读取文件；否则使用下方的字符串。
REF_TEXT_DEFAULT = "这里填入你的参考音频ref.wav实际上说的内容。"

# 4. 模型检查点路径 (默认为 OpenAudio-S1-Mini)
CHECKPOINT_DIR = "checkpoints/openaudio-s1-mini"
# ==========================================================

def get_python_exec():
    """获取当前运行环境的 Python 解释器路径"""
    return sys.executable

def check_files():
    """检查必要的文件是否存在"""
    if not os.path.exists("fish_speech"):
        print("❌ 错误: 请将此脚本放在 fish-speech 仓库的根目录下运行。")
        sys.exit(1)
    
    if not os.path.exists(REF_AUDIO_PATH):
        print(f"❌ 错误: 未找到参考音频 '{REF_AUDIO_PATH}'。")
        print("👉 请在根目录下放入一个 .wav 音频文件，并命名为 ref.wav")
        sys.exit(1)

def get_ref_text():
    """获取参考文本"""
    if os.path.exists("ref.txt"):
        with open("ref.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    return REF_TEXT_DEFAULT

def download_model():
    """如果模型不存在，尝试自动下载"""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"⬇️ 未检测到模型，正在下载 OpenAudio-S1-Mini 到 {CHECKPOINT_DIR} ...")
        try:
            # 需要安装 huggingface_hub[cli]
            subprocess.run([
                "huggingface-cli", "download", "fishaudio/openaudio-s1-mini",
                "--local-dir", CHECKPOINT_DIR
            ], check=True)
            print("✅ 模型下载完成。")
        except subprocess.CalledProcessError:
            print("❌ 下载失败。请确保安装了 huggingface_hub: pip install huggingface_hub[cli]")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ 未找到 huggingface-cli 命令，请先安装: pip install huggingface_hub[cli]")
            sys.exit(1)
    else:
        print(f"✅ 检测到模型文件: {CHECKPOINT_DIR}")

def run_step_1_encode(ref_audio):
    """步骤 1: 将参考音频编码为 VQ Token (fake.npy)"""
    print("\n🔹 Step 1/3: 正在编码参考音频 (VQGAN Encode)...")
    
    # 注意：根据新版代码结构，路径通常在 fish_speech/models/dac/inference.py
    script_path = "fish_speech/models/dac/inference.py"
    
    # 检测设备 (macOS M系列芯片使用mps，其他使用cuda或cpu)
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    cmd = [
        get_python_exec(), script_path,
        "-i", ref_audio,
        "--checkpoint-path", os.path.join(CHECKPOINT_DIR, "codec.pth"),
        "--device", device
    ]
    
    try:
        subprocess.run(cmd, check=True)
        if not os.path.exists("fake.npy"):
            raise FileNotFoundError("fake.npy 未生成")
        print("✅ 参考音频编码成功 (fake.npy)")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        sys.exit(1)

def run_step_2_generate(target_text, ref_text):
    """步骤 2: 从文本生成语义 Token (LLM Generate)"""
    print("\n🔹 Step 2/3: 正在生成语义 Token (LLM Text2Semantic)...")
    print(f"   参考文本: {ref_text}")
    print(f"   目标文本: {target_text}")

    script_path = "fish_speech/models/text2semantic/inference.py"
    
    # 检测设备 (macOS M系列芯片使用mps，其他使用cuda或cpu)
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    cmd = [
        get_python_exec(), script_path,
        "--text", target_text,
        "--prompt-text", ref_text,
        "--prompt-tokens", "fake.npy",
        "--checkpoint-path", CHECKPOINT_DIR,
        "--device", device,
        # "--compile" # 如果是 Linux 且有 Triton，可以取消注释加速，Windows下建议关闭
    ]

    try:
        subprocess.run(cmd, check=True)
        # 检查文件是否在 temp 目录下生成
        if os.path.exists("temp/codes_0.npy"):
            import shutil
            shutil.move("temp/codes_0.npy", "codes_0.npy")
            print("✅ 语义 Token 生成成功 (codes_0.npy)")
        elif os.path.exists("codes_0.npy"):
            print("✅ 语义 Token 生成成功 (codes_0.npy)")
        else:
            raise FileNotFoundError("codes_0.npy 未生成")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        sys.exit(1)

def run_step_3_decode():
    """步骤 3: 将语义 Token 解码为音频 (VQGAN Decode)"""
    print("\n🔹 Step 3/3: 正在解码为音频 (VQGAN Decode)...")
    
    script_path = "fish_speech/models/dac/inference.py"
    
    # 检测设备 (macOS M系列芯片使用mps，其他使用cuda或cpu)
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    cmd = [
        get_python_exec(), script_path,
        "-i", "codes_0.npy",
        "--checkpoint-path", os.path.join(CHECKPOINT_DIR, "codec.pth"),
        "--device", device
    ]

    try:
        subprocess.run(cmd, check=True)
        # 输出文件通常是 codes_0.wav
        if os.path.exists("codes_0.wav"):
            print(f"\n🎉 成功！音频已保存为: {os.path.abspath('codes_0.wav')}")
        elif os.path.exists("fake.wav"):
            print(f"\n🎉 成功！音频已保存为: {os.path.abspath('fake.wav')}")
        else:
            print("❌ 未找到生成的音频文件。")
    except Exception as e:
        print(f"❌ 解码失败: {e}")
        sys.exit(1)

def cleanup():
    """清理临时文件"""
    for file in ["fake.npy", "codes_0.npy", "fake.wav"]:
        if os.path.exists(file):
            try:
                # os.remove(file) # 如果你想保留中间文件用于调试，请注释掉这一行
                pass
            except:
                pass

def main():
    print("🐟 Fish-Speech 简易推理脚本启动...")
    
    # 0. 准备工作
    check_files()
    ref_text = get_ref_text()
    
    # 检查是否有参考文本，如果使用的是默认文本且没改，给予提示
    if ref_text == "这里填入参考音频ref.wav实际上说的内容。" and not os.path.exists("ref.txt"):
        print("⚠️  警告: 你正在使用默认的参考文本。为了获得最佳克隆效果，")
        print("   请编辑脚本中的 REF_TEXT_DEFAULT 或创建 ref.txt 文件，填入 ref.wav 的真实内容。")
        input("   按 Enter 继续，或 Ctrl+C 退出修改...")

    download_model()

    # 1. 编码参考音频 -> fake.npy
    run_step_1_encode(REF_AUDIO_PATH)

    # 2. 文本生成语义 -> codes_0.npy
    run_step_2_generate(TARGET_TEXT, ref_text)

    # 3. 语义解码音频 -> codes_0.wav
    run_step_3_decode()

    # cleanup() # 可选：清理中间文件

if __name__ == "__main__":
    main()