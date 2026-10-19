import argparse
import json
import logging
import os
import shutil
import sys
import time
import subprocess

# ---- 進捗表示 ＆ macOS通知ヘルパー関数 ----
def set_terminal_title(title: str):
    """ターミナルのウィンドウタイトルを進捗に合わせて動的に更新する"""
    try:
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    except Exception:
        pass

def send_macos_notification(title: str, message: str, sound: str = "Glass"):
    """macOSネイティブのデスクトップ通知を音付きで送信する"""
    try:
        clean_title = title.replace('"', '\\"')
        clean_msg = message.replace('"', '\\"')
        script = f'display notification "{clean_msg}" with title "{clean_title}" sound name "{sound}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        pass

def print_step_banner(step_num: int, total_steps: int, title: str, description: str = ""):
    """見やすい日本語ステップバナーをターミナルに表示し、タイトルも更新する"""
    percent = int((step_num - 1) / total_steps * 100)
    set_terminal_title(f"[{percent}%] ({step_num}/{total_steps}) {title} - AutoComposer Upscaler")
    bar = "=" * 62
    print(f"\n{bar}")
    print(f" ⚡【工程 {step_num}/{total_steps}】 {title}  (進捗目安: {percent}%)")
    if description:
        print(f"    ℹ️  {description}")
    print(f"{bar}\n")

# ACE-Step Paths
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure ACE-Step modules are reachable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acestep.inference import (
    GenerationConfig,
    GenerationParams,
    generate_music,
)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_json_metadata(json_path: str):
    """Load metadata from the given JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="ACE-Step AutoComposer Upscaler (Turbo -> SFT)")
    parser.add_argument("--json", type=str, required=True, help="Path to the .json metadata file to upscale")
    parser.add_argument("--format", type=str, default="flac", choices=["flac", "mp3", "wav", "wav32"], help="Output audio format")
    args = parser.parse_args()

    t_start = time.time()
    
    set_terminal_title("[0%] 準備中 - AutoComposer Upscaler")
    send_macos_notification("⚡ AutoComposer Upscaler", "高音質アップスケール処理を開始しました。", sound="Hero")
    
    # ---- 1. メタデータの読み込み ----
    print_step_banner(1, 2, "📄 設計図の解析 ＆ SFTモデルの準備", "元の楽曲のシード値・プロンプト・構成コードを読み込み、SFTモデルを初期化しています...")
    logger.info(f"📄 Loading metadata from: {args.json}")
    try:
        metadata = load_json_metadata(args.json)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        sys.exit(1)
        
    caption = metadata.get("caption", "")
    lyrics_hiragana = metadata.get("lyrics", "")
    vocal_language = metadata.get("vocal_language", "ja")
    bpm = metadata.get("bpm", "")
    keyscale = metadata.get("keyscale", "")
    timesignature = metadata.get("timesignature", "")
    duration = metadata.get("duration", -1.0)
    audio_codes = metadata.get("audio_codes", "")
    seed = metadata.get("seed", -1)

    if not audio_codes:
        logger.warning("⚠️ audio_codesがJSONに含まれていません。Thinking=Trueでコード進行・メロディを新規構成し、SFT 50ステップで高品質レンダリングします。")
        use_thinking = True
    else:
        logger.info("✨ 元の楽曲の設計図トークン (audio_codes) を検出しました。メロディ・歌唱を完全維持したままSFT 50ステップで高音質化します。")
        use_thinking = False

    # ---- 2. モデルの初期化 ----
    logger.info("🚀 ACE-Step SFTモデルを初期化しています (Apple Silicon 最適化)...")
    dit_handler = AceStepHandler()
    s_msg, s_success = dit_handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path="acestep-v15-xl-sft",
        device="auto",
        backend="mlx"
    )
    if not s_success:
        logger.error(f"❌ SFTモデルの初期化に失敗しました: {s_msg}")
        sys.exit(1)
        
    logger.info("🚀 4B 言語モデル (LLM) を初期化しています...")
    llm_handler = LLMHandler()
    s_msg, s_success = llm_handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        checkpoint_dir=os.path.join(str(PROJECT_ROOT), "checkpoints"),
        lm_model_path="acestep-5Hz-lm-4B",
        device="auto",
        backend="mlx"
    )
    if not s_success:
        logger.error(f"❌ LLMの初期化に失敗しました: {s_msg}")
        sys.exit(1)

    # ---- 3. アップスケール用パラメータの構築 ----
    # Upscale専用: audio_codesがある場合はthinking=False (既存のコードをそのまま使う), inference_steps=50, shift=3.0, dcw=False
    params = GenerationParams(
        task_type="text2music",
        thinking=use_thinking,
        audio_codes=audio_codes if not use_thinking else "",
        caption=caption,
        lyrics=lyrics_hiragana,
        vocal_language=vocal_language,
        duration=duration,
        keyscale=keyscale,
        bpm=bpm,
        timesignature=timesignature,
        fade_out_duration=0.0,
        inference_steps=50, # SFT用の高解像度ステップ
        guidance_scale=7.0,
        use_adg=False,
        sampler_mode="euler",
        shift=3.0,          # SFT必須パラメータ
        dcw_enabled=True,   # 修正: SFTでもノイズを防ぐためにDCWを有効化
        seed=seed,          # 全く同じシード値を指定
    )
    
    config = GenerationConfig(
        batch_size=1,
        audio_format=args.format,
        use_random_seed=False, # 確実に同じシードを使う
    )
    
    temp_save_dir = os.path.join(str(PROJECT_ROOT), "gradio_outputs", "autochord_temp_upscale")
    os.makedirs(temp_save_dir, exist_ok=True)
    
    # ---- 4. 生成の実行 ----
    print_step_banner(2, 2, "🎼 高音質アップスケール生成中 (50 Steps SFT)", "同じメロディのまま、高精細な本番品質（SFTモデル）で楽曲を再レンダリングしています...")
    logger.info("🎵 Starting high-quality SFT Upscale generation...")
    result = generate_music(
        dit_handler,
        llm_handler,
        params=params,
        config=config,
        save_dir=temp_save_dir,
    )
    
    # ---- 5. 出力ファイルの保存 ----
    desktop_dir = os.path.expanduser("~/Desktop")
    if result.success and result.audios:
        logger.info("✅ Upscale completed successfully!")
        for idx, audio in enumerate(result.audios):
            temp_path = audio.get("path")
            if temp_path and os.path.exists(temp_path):
                # 元のファイル名を推測して _Upscaled をつける
                base_name = os.path.splitext(os.path.basename(args.json))[0]
                final_name = f"{base_name}_Upscaled.{args.format}"
                dest_path = os.path.join(desktop_dir, final_name)
                
                shutil.copy(temp_path, dest_path)
                logger.info(f"🎉 Saved UPSCALED track to Desktop: {dest_path}")
                
                try:
                    import platform
                    if platform.system() == "Darwin":
                        # 自動再生
                        play_script = (
                            f"osascript "
                            f"-e 'tell application \"QuickTime Player\" to activate' "
                            f"-e 'tell application \"QuickTime Player\" to open POSIX file \"{dest_path}\"' "
                            f"-e 'tell application \"QuickTime Player\" to play document 1'"
                        )
                        os.system(play_script)
                except Exception:
                    pass
                
        shutil.rmtree(temp_save_dir, ignore_errors=True)
        set_terminal_title("[100%] 🎉 アップスケール完了！ - AutoComposer Upscaler")
        send_macos_notification("🎉 AutoComposer Upscaler", "高音質アップスケールが完了しました！デスクトップを確認してください。", sound="Glass")
        print("\n" + "★" * 62)
        print(" 🎉 高音質アップスケール（SFT）が正常に完了しました！")
        print(f" ⏱️  総所要時間: {time.time() - t_start:.1f} 秒")
        print(f" 📁 保存先: デスクトップ")
        print("★" * 62 + "\n")
    else:
        logger.error(f"Failed to upscale music: {result.status_message}")
        shutil.rmtree(temp_save_dir, ignore_errors=True)
        sys.exit(1)
        
    logger.info(f"Total time elapsed: {time.time() - t_start:.1f} seconds.")

if __name__ == "__main__":
    main()
