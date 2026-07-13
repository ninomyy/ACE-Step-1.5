import argparse
import json
import logging
import os
import shutil
import sys
import time

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
from acestep.ui.gradio.events.generation.llm_init import init_llm
from acestep.ui.gradio.events.generation.model_init import init_dit

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
    
    # ---- 1. メタデータの読み込み ----
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
        logger.error("❌ audio_codes not found in JSON. Cannot perform precise upscale without audio codes!")
        sys.exit(1)

    # ---- 2. モデルの初期化 ----
    logger.info("🚀 Initializing ACE-Step SFT Model...")
    dit_handler = init_dit(None)
    
    # 念のため SFT モデルかチェック
    model_id = getattr(dit_handler, 'model_id', '').lower()
    if "sft" not in model_id:
        logger.warning(f"⚠️ Warning: The currently loaded model ({model_id}) does not appear to be an SFT model.")
        logger.warning("Upscaling will still run with SFT parameters, but quality may vary if an SFT model is not loaded.")
        
    llm_handler = init_llm(None)
    if not llm_handler.llm_initialized:
        logger.error("❌ Failed to initialize LLM. Please make sure LM path is configured.")
        sys.exit(1)

    # ---- 3. アップスケール用パラメータの構築 ----
    # Upscale専用: thinking=False (既存のaudio_codesをそのまま使う), inference_steps=50, shift=3.0, dcw=False
    params = GenerationParams(
        task_type="text2music",
        thinking=False, # 既存のAudio Codesを強制使用する
        audio_codes=audio_codes,
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
    else:
        logger.error(f"Failed to upscale music: {result.status_message}")
        shutil.rmtree(temp_save_dir, ignore_errors=True)
        sys.exit(1)
        
    logger.info(f"Total time elapsed: {time.time() - t_start:.1f} seconds.")

if __name__ == "__main__":
    main()
