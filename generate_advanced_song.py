#!/usr/bin/env python3
"""
Fully Automated Song Writer and Composer (Ollama 32B + ACE-Step 1.5 XL-turbo)
This script runs entirely offline on local Mac, utilizing:
1. Ollama (qwen2.5:32b) to write beautiful Japanese lyrics and generate English music captions.
2. Memory release hack to immediately unload the 32B model from unified memory.
3. ACE-Step 1.5 (4B LM + XL-turbo) to compose and generate high-quality FLAC/WAV songs.
4. Saves final outputs directly to the user's Desktop.
"""

import os
import sys
import json
import time
import argparse
import requests
import shutil
from pathlib import Path

# Increase ACE-Step internal generation timeout to 1 hour (3600s) to accommodate Heun sampler and 50 steps
os.environ["ACESTEP_GENERATION_TIMEOUT"] = "3600"

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3.6:27b"

def parse_lyrics_file(file_path):
    """Parse theme, caption, lyrics, and lyrics_hiragana from a generated lyrics file."""
    logger.info(f"既存の歌詞ファイルを読み込んで解析しています: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Clean carriage returns for standard splitting
    content = content.replace("\r\n", "\n")
    
    theme = ""
    # Parse Theme
    import re
    theme_match = re.search(r"^\s*Theme:\s*(.+)$", content, re.MULTILINE)
    if theme_match:
        theme = theme_match.group(1).strip()
        
    estimated_duration = 0
    # Parse Target Duration
    duration_match = re.search(r"^\s*Target Duration:\s*(\d+)s", content, re.MULTILINE)
    if duration_match:
        estimated_duration = int(duration_match.group(1))
        
    # Split by sections
    lyrics_section = ""
    lyrics_hira_section = ""
    caption_section = ""
    
    try:
        if "■ 歌詞（漢字かな混じり - 通常表示用）" in content:
            parts = content.split("■ 歌詞（漢字かな混じり - 通常表示用）")
            if len(parts) > 1:
                sub_content = parts[1]
                if "-----------------------------------------------------" in sub_content:
                    sub_content = sub_content.split("-----------------------------------------------------", 1)[1]
                if "■" in sub_content:
                    lyrics_section = sub_content.split("■", 1)[0].strip()
                else:
                    lyrics_section = sub_content.strip()
                    
        if "■ 歌詞（ひらがな - ACE-Step 歌唱入力用）" in content:
            parts = content.split("■ 歌詞（ひらがな - ACE-Step 歌唱入力用）")
            if len(parts) > 1:
                sub_content = parts[1]
                if "-----------------------------------------------------" in sub_content:
                    sub_content = sub_content.split("-----------------------------------------------------", 1)[1]
                if "■" in sub_content:
                    lyrics_hira_section = sub_content.split("■", 1)[0].strip()
                else:
                    lyrics_hira_section = sub_content.strip()
                    
        if "■ 音楽構成プロンプト (Music Caption)" in content:
            parts = content.split("■ 音楽構成プロンプト (Music Caption)")
            if len(parts) > 1:
                sub_content = parts[1]
                if "-----------------------------------------------------" in sub_content:
                    sub_content = sub_content.split("-----------------------------------------------------", 1)[1]
                caption_section = sub_content.strip()
    except Exception as parse_err:
        logger.error(f"Error parsing lyrics file sections: {str(parse_err)}")
        
    return theme, caption_section, lyrics_section, lyrics_hira_section, estimated_duration

def get_lyrics_and_caption_from_ollama(theme, duration, model_name=DEFAULT_MODEL):
    """Call Ollama API to generate lyrics and caption based on the theme."""
    logger.info(f"Ollama ({model_name}) を使用して、歌詞、ひらがな歌詞、および音楽構成プロンプトを生成しています...")
    
    if duration == 0:
        duration_instruction = """The target song duration is set to dynamic mode. You MUST write a FULL-LENGTH song (3 to 5 minutes). 
To achieve this, you MUST include a complete and long structure:
[Intro]
[Verse 1]
[Pre-Chorus]
[Chorus]
[Verse 2]
[Pre-Chorus]
[Chorus]
[Bridge]
[Guitar Solo] or [Instrumental Break]
[Final Chorus]
[Outro]

CRITICAL: Do NOT write a short song. You must write at least 30 to 40 lines of lyrics. Do not abbreviate."""
    else:
        duration_instruction = f"The target song duration is strictly set to {duration} seconds. Adjust the length and pacing of the lyrics so that the vocal performance and structure fit comfortably within this duration."

    prompt = f"""
You are an expert Music Producer and Songwriter for Japanese City Pop.
Based on the theme below, generate a beautiful Japanese lyric for a female vocalist, and a matching English Music Caption for ACE-Step 1.5.

Theme: {theme}
{duration_instruction}

Rules:
1. For [Music Caption], write in English. To prevent AI prompt overload, keep it concise (maximum 30-50 words, using 5-8 comma-separated tags). It MUST include specific technical tags to ensure highest musical quality:
   - "studio quality, high-fidelity, pristine audio, professional mixing, 48kHz, lossless"
   - Specify vocal style clearly (e.g., "breathy whisper voice", "emotional falsetto", "powerful belting", "clear female vocal").
   - CRITICAL: If the user specifies a BPM, Key, or Time Signature in the Theme, you MUST use those EXACT values in the English caption. DO NOT change them.
   - Describe the genre (City Pop) and instruments. Limit instrument specification to 1-2 main instruments (e.g., "piano, grooving bass") to avoid oversaturation.
   - Crucially, analyze the theme and determine the ending style. Describe this chosen ending clearly in English at the end of the [Music Caption] (e.g., "ends with a smooth 15-second instrumental fade-out" or "concludes with a clean, dramatic final piano chord").
2. For [Lyrics], write in Japanese (Kanji/Kana mix). To achieve a dramatic J-Pop/City Pop structure, use THESE specific structural tags in square brackets:
   [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Instrumental Break], [Quiet Chorus], [Final Chorus], [Outro].
   You may optionally add brief stylistic or instrumental cues inside the tags (e.g., [Chorus: energetic brass]) to dynamically change the mood, but keep them musical and effective. Ensure the lyrics are poetic, rhythmic, and fit the city pop melody.
3. For [Lyrics Hiragana], write the EXACT SAME lyrics as in Rule 2, but converted entirely into Hiragana (ひらがな) and Katakana (for loan words) to prevent the vocal synthesis model from mispronouncing the words.
   - Do NOT convert the structural tags (like [Intro], [Chorus]) into hiragana—keep them in English brackets.
   - Crucially, verify all Japanese dictionary readings for absolute correctness and avoid common pronunciation/spelling traps:
     * "遠く" must be spelled as "とおく" (NOT "とうく")
     * "通り" must be spelled as "とおり" (NOT "とうり")
     * "多く" must be spelled as "おおく" (NOT "おうく")
     * "雨滴" must be read as "うてき" (NOT "あめてき")
     * "雨音" must be read as "あまおと" or "あめおと" (NOT "あめのおと")
     * "静けさ" must be read as "しずけさ" (NOT "しずかさ")
     * "耳を澄ませて" must be read as "みみをすませて" (do not omit "み")
   - Double check that the Hiragana version matches the Kanji/Kana mixed version syllable-by-syllable without omitting any words or characters.
4. You are allowed to output your thought process or reasoning before the JSON. Take your time to carefully write out a full 3 to 5-minute song. After your reasoning, you MUST output a JSON block containing three keys: "caption", "lyrics", and "lyrics_hiragana".

Response JSON Format:
{{
  "caption": "(English caption here)",
  "lyrics": "(Japanese lyrics in Kanji/Kana mix here)",
  "lyrics_hiragana": "(Japanese lyrics in Hiragana/Katakana here)"
}}
"""

    try:
        # Increase timeout to 300s to accommodate long reasoning/thinking outputs of Qwen 3.6.
        # Remove format="json" to prevent Ollama from forcing JSON output constraints during CoT thinking.
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            },
            timeout=600
        )
        response.raise_for_status()
        result_json = response.json()
        
        # Fallback to 'thinking' field if Ollama misroutes the final response
        raw_response = result_json.get("response", "").strip()
        if not raw_response:
            raw_response = result_json.get("thinking", "").strip()
            
        # Robustly extract JSON using regex (handles markdown, <think> blocks, etc.)
        import re
        parsed = {}
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except Exception as parse_err:
                logger.warning(f"Failed to parse regex matched JSON: {parse_err}")
                
        if not parsed:
            # Fallback direct load
            parsed = json.loads(raw_response)
            
        caption = parsed.get("caption", "").strip()
        lyrics = parsed.get("lyrics", "").strip()
        lyrics_hiragana = parsed.get("lyrics_hiragana", "").strip()
        
        if not caption or not lyrics or not lyrics_hiragana:
            raise ValueError("Ollama returned empty caption, lyrics, or lyrics_hiragana.")
            
        return caption, lyrics, lyrics_hiragana
    except Exception as e:
        logger.error(f"Error calling Ollama: {str(e)}")
        # Fallback prompts if Ollama fails
        logger.warning("Using pre-defined fallback J-Pop / City Pop values.")
        caption = "A beautiful and melodic Japanese city pop song, clear and emotional female vocal, retro synthesizer chords, grooving bassline, nostalgic 80s urban atmosphere, ending with a smooth instrumental fade-out. Sung in Japanese, high quality, 110 BPM, key of A major."
        lyrics = "[Intro]\n[Verse 1]\nビルが立ち並ぶ 街 of ネオン\n君の影を 探しているの\n[Chorus]\n真夜中のハイウェイ 走り抜け\n二人だけの世界へ 連れてって\n踊ろうよ 朝が来るまで\n[Outro]"
        lyrics_hiragana = "[Intro]\n[Verse 1]\nびるがたちならぶ まちのねおん\nきみのかげを ささがしているの\n[Chorus]\nまよなかのはいうぇい はしりぬけ\nふたりだけのせかいへ つれてって\nおどろうよ あさがくるまで\n[Outro]"
        return caption, lyrics, lyrics_hiragana

def unload_ollama_model(model_name=DEFAULT_MODEL):
    """Force unload the model from GPU/Unified Memory immediately by setting keep_alive to 0."""
    logger.info(f"Macのメモリからモデル {model_name} を解放しています...")
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_name,
                "prompt": "",
                "keep_alive": 0,
                "stream": False
            },
            timeout=10
        )
        response.raise_for_status()
        logger.info("Ollamaのメモリ解放が完了しました。")
    except Exception as e:
        logger.warning(f"Ollamaモデルの強制アンロードに失敗しました: {str(e)}")

def detect_keyscale(theme, caption):
    """
    Detect the musical keyscale from the Japanese theme or the English caption.
    Returns standard format like 'C Major', 'A Minor', or '' if not found.
    """
    import re
    # Helper to normalize accidentals
    def normalize_acc(acc_str):
        if not acc_str:
            return ""
        a = acc_str.lower()
        if a in ['#', '♯', 'シャープ', 'sharp']:
            return "#"
        if a in ['b', '♭', 'フラット', 'flat']:
            return "b"
        return ""

    # 1. Try extracting from Japanese theme first (user explicit choice)
    if theme:
        # Match pattern: Note + optional hyphen/space + optional accidental + mode
        # Accidental 'b' and mode 'm' must not be followed by other english letters (using lookahead)
        matches = re.finditer(r'([A-G])\s*[-]?\s*(#|b(?![a-z])|♯|♭|シャープ|フラット)?\s*(major|minor|Major|Minor|メジャー|マイナー|m(?![a-z]))?', theme, re.IGNORECASE)
        for m in matches:
            note = m.group(1).upper()
            acc_raw = m.group(2)
            mode_str = m.group(3)
            
            acc = normalize_acc(acc_raw)
            
            is_minor = False
            if mode_str:
                ms = mode_str.lower()
                if ms in ['minor', 'マイナー', 'm']:
                    is_minor = True
            
            # Check context to verify it's a key, not a random character
            start_idx = m.start()
            end_idx = m.end()
            context_before = theme[max(0, start_idx - 10):start_idx]
            context_after = theme[end_idx:min(len(theme), end_idx + 10)]
            
            has_key_context = (
                mode_str is not None or
                acc_raw is not None or
                "キー" in context_before or "調" in context_after or
                "key" in context_before.lower() or "で" in context_after
            )
            
            if has_key_context:
                mode = "Minor" if is_minor else "Major"
                return f"{note}{acc} {mode}"

    # 2. Try extracting from English caption (Ollama generated key)
    if caption:
        # Match "key of/in/key:" followed by note [A-G] and optional flat/sharp/mode
        match = re.search(r'\b(?:key of|in|key\s*:\s*)\s*([A-G])\s*[-]?\s*(sharp|flat|#|b(?![a-z])|♯|♭)?\s*(major|minor|m(?![a-z]))?\b', caption, re.IGNORECASE)
        if match:
            note = match.group(1).upper()
            acc_str = match.group(2)
            mode_str = match.group(3)
            
            acc = normalize_acc(acc_str)
                    
            is_minor = False
            if mode_str:
                m = mode_str.lower()
                if m in ['minor', 'm']:
                    is_minor = True
                    
            mode = "Minor" if is_minor else "Major"
            return f"{note}{acc} {mode}"
            
    return ""

def detect_bpm(theme, caption):
    """Detect BPM from theme or caption."""
    import re
    text = f"{theme} {caption}"
    match = re.search(r'\b(?:bpm|テンポ|tempo)\s*[:=]?\s*(\d{2,3})\b|\b(\d{2,3})\s*(?:bpm|テンポ|tempo)\b', text, re.IGNORECASE)
    if match:
        val = match.group(1) or match.group(2)
        return int(val)
    return None

def detect_timesignature(theme, caption):
    """Detect time signature from theme or caption."""
    import re
    text = f"{theme} {caption}"
    match = re.search(r'\b(3/4|4/4|6/8)(?:\s*(?:time|拍子))?\b', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def sanitize_caption(theme, caption):
    """Ensure the caption uses the exact BPM and Key from the theme if specified, overriding LLM hallucinations."""
    import re
    
    theme_key = detect_keyscale(theme, "")
    theme_bpm = detect_bpm(theme, "")
    
    sanitized = caption
    
    if theme_key:
        key_pattern = r'\b(key\s+of\s+|in\s+key\s+of\s+|in\s+|key:\s*)?([A-G]\s*[-]?\s*(?:sharp|flat|#|b|♯|♭)?\s*(?:major|minor|m)(?![a-z]))\b'
        
        def replace_key(match):
            prefix = match.group(1) or "key of "
            return f"{prefix}{theme_key}"
            
        if re.search(key_pattern, sanitized, re.IGNORECASE):
            sanitized = re.sub(key_pattern, replace_key, sanitized, flags=re.IGNORECASE)
        else:
            sanitized += f", key of {theme_key}"
            
    if theme_bpm:
        bpm_pattern = r'\b(\d{2,3})\s*(?:bpm)\b|\b(?:bpm)\s*(\d{2,3})\b'
        
        def replace_bpm(match):
            if match.group(1):
                return f"{theme_bpm} BPM"
            return f"BPM {theme_bpm}"
            
        if re.search(bpm_pattern, sanitized, re.IGNORECASE):
            sanitized = re.sub(bpm_pattern, replace_bpm, sanitized, flags=re.IGNORECASE)
        else:
            sanitized += f", {theme_bpm} BPM"
            
    return sanitized

def calculate_estimated_duration(lyrics_hiragana):
    """Calculate an optimal song duration (seconds) based on the volume of lyrics and structural tags."""
    import re
    # Extract tags
    tags = re.findall(r'\[.*?\]', lyrics_hiragana)
    unique_tags = set(t.lower() for t in tags)
    
    instrumental_time = 0
    if "[intro]" in unique_tags: instrumental_time += 15
    if "[outro]" in unique_tags: instrumental_time += 15
    if "[instrumental break]" in unique_tags or "[guitar solo]" in unique_tags: instrumental_time += 15
    
    if instrumental_time == 0:
        instrumental_time = 20
        
    # Strip tags, spaces, punctuation to count pure syllables
    clean_lyrics = re.sub(r'\[.*?\]', '', lyrics_hiragana)
    clean_lyrics = re.sub(r'[\s　、。！？,.\!\?]', '', clean_lyrics)
    
    char_count = len(clean_lyrics)
    
    # Approx 3.5 syllables per second for J-Pop
    vocal_time = int(char_count / 3.5)
    
    # Calculate pauses between lines
    lines = len([line for line in lyrics_hiragana.split('\n') if line.strip() and not line.strip().startswith('[')])
    pause_time = int(lines * 1.5) # 1.5s pause per lyric line
    
    total_time = instrumental_time + vocal_time + pause_time
    
    # Constrain to 30s - 210s
    # 長時間（3分半以上）を一気に生成しようとするとDiTのコンテキスト限界を超えて
    # 曲の後半で同じ音がループする現象(Repetition Collapse)が起きるため、安全限界(210秒)を設ける
    return max(30, min(total_time, 210))

def get_total_ram_gb():
    """Check system RAM in GB to determine if parallel model loading is safe."""
    try:
        import subprocess
        out = subprocess.check_output(['sysctl', 'hw.memsize'])
        mem_bytes = int(out.decode().strip().split(':')[1].strip())
        return mem_bytes / (1024**3)
    except Exception:
        return 32.0

def wait_for_ollama_unload():
    """Poll Ollama API to confirm models are unloaded, eliminating fixed sleep times."""
    logger.info("Ollamaのメモリ解放を待機しています...")
    for _ in range(30):
        try:
            resp = requests.get("http://127.0.0.1:11434/api/ps", timeout=1)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if len(models) == 0:
                    logger.info("✅ Ollamaのメモリが即座に解放されました。")
                    return
        except Exception:
            pass
        time.sleep(0.1)
    logger.warning("Ollamaのメモリ解放の確認がタイムアウトしましたが、このまま続行します。")

def main():
    parser = argparse.ArgumentParser(description="Auto-Compose song using Ollama + ACE-Step 1.5 XL")
    parser.add_argument("--theme", type=str, required=True, help="Theme for the song (in Japanese) or path to an existing Lyrics_*.txt file")
    parser.add_argument("--duration", type=int, default=0, help="Duration of the generated song in seconds (0 = dynamic based on lyrics)")
    parser.add_argument("--format", type=str, default="flac", choices=["flac", "wav", "mp3"], help="Output audio format")
    parser.add_argument("--bit_depth", type=int, default=16, choices=[16, 24, 32], help="Audio bit depth for WAV/FLAC")
    parser.add_argument("--model_type", type=str, default="sft", choices=["sft", "turbo"], help="ACE-Step 1.5 model to use")
    parser.add_argument("--ollama_model", type=str, default=DEFAULT_MODEL, help="Ollama model to use")
    
    args = parser.parse_args()
    
    t_start = time.time()
    
    total_ram_gb = get_total_ram_gb()
    logger.info(f"システムのRAM容量を確認しました: {total_ram_gb:.1f} GB")
    
    # Check if args.theme is a path to an existing lyrics file
    lyrics_file_path = None
    if os.path.exists(args.theme):
        lyrics_file_path = args.theme
    else:
        # Check Desktop folder
        desktop_path = os.path.expanduser(f"~/Desktop/{args.theme}")
        if os.path.exists(desktop_path) and args.theme.endswith(".txt"):
            lyrics_file_path = desktop_path
            
    is_retry = False
    if lyrics_file_path:
        # ---- Retry / Re-generation Mode ----
        logger.info("=====================================================")
        logger.info("🔄 再生成モード: 既存の歌詞とキャプションを再利用します")
        logger.info(f"   読み込み元ファイル: {lyrics_file_path}")
        logger.info("=====================================================")
        
        parsed_theme, caption, lyrics, lyrics_hiragana, estimated_duration = parse_lyrics_file(lyrics_file_path)
        if parsed_theme and caption and lyrics_hiragana:
            is_retry = True
            # Override theme to keep the original theme name for file outputs
            args.theme = parsed_theme
            logger.info(f"解析されたテーマ: {args.theme}")
            
            # LLMの幻覚（異なるBPMやKeyの出力）をユーザー指定値で強制上書き
            caption = sanitize_caption(args.theme, caption)
            
            logger.info("歌詞とキャプションの抽出に成功しました。Ollamaでの生成をスキップします。")
        else:
            logger.error("Failed to parse lyrics file contents properly. Falling back to normal mode.")
            
    def init_models():
        config_path = "acestep-v15-xl-sft" if args.model_type == "sft" else "acestep-v15-xl-turbo"
        model_display_name = "SFT" if args.model_type == "sft" else "Turbo"
        
        logger.info(f"🎵 ACE-Step 1.5 XL {model_display_name} モデルを初期化しています (AIが作曲する準備をしています)...")
        d_handler = AceStepHandler()
        s_msg, s_success = d_handler.initialize_service(
            project_root=str(PROJECT_ROOT),
            config_path=config_path,
            device="auto",
            offload_to_cpu=False,
        )
        if not s_success:
            logger.error(f"作曲AIモデルの初期化に失敗しました: {s_msg}")
            sys.exit(1)
            
        logger.info("🧠 ACE-Step 1.5 言語モデル (LM) を初期化しています...")
        l_handler = LLMHandler()
        s_msg, s_success = l_handler.initialize(
            checkpoint_dir=os.path.join(str(PROJECT_ROOT), "checkpoints"),
            lm_model_path="acestep-5Hz-lm-4B",
            device="auto",
            backend="mlx"
        )
        if not s_success:
            logger.error(f"LLM initialization failed: {s_msg}")
            sys.exit(1)
        return d_handler, l_handler

    if not is_retry:
        # ---- 1. Run Ollama 作詞 & プロンプト生成 ----
        caption, lyrics, lyrics_hiragana = get_lyrics_and_caption_from_ollama(args.theme, args.duration, args.ollama_model)
        
        # LLMの幻覚（異なるBPMやKeyの出力）をユーザー指定値で強制上書き
        caption = sanitize_caption(args.theme, caption)
        
        # アルゴリズムで歌詞の文字数とタグから最適な曲の長さを物理計算する（LLMの幻覚を排除）
        estimated_duration = calculate_estimated_duration(lyrics_hiragana)
        logger.info(f"✨ 歌詞の分量から計算された最適な楽曲の長さ: {estimated_duration}秒")
        
        logger.info("\n" + "="*50)
        logger.info("生成された音楽構成プロンプト (英語):")
        logger.info(caption)
        logger.info("-"*50)
        logger.info("生成された歌詞 (日本語):")
        logger.info(lyrics)
        logger.info("-"*50)
        logger.info("生成された歌唱用ひらがな歌詞:")
        logger.info(lyrics_hiragana)
        logger.info("="*50 + "\n")
        
        # ---- 2. Ollamaのメモリを即座に解放 (アンロードハック) ----
        unload_ollama_model(args.ollama_model)
        
        # フラグ立てと同義の即時ポーリングで待ち時間を最小化
        wait_for_ollama_unload()
        
    # ---- 3. ACE-Step 1.5 XL-turbo + 4B LM で作曲とレンダリング ----
    dit_handler, llm_handler = init_models()
        
    logger.info(f"🎼 楽曲のレンダリング（生成）を開始します: テーマ='{args.theme}', 長さ={args.duration}秒, フォーマット={args.format}...")
    
    # Detect keyscale, BPM, and Time Signature from theme or caption
    detected_key = detect_keyscale(args.theme, caption)
    detected_bpm = detect_bpm(args.theme, caption)
    detected_ts = detect_timesignature(args.theme, caption)
    
    # Detect fade-out intention
    fade_out_duration = 0.0
    text_to_check = f"{caption} {lyrics}".lower()
    if "fade-out" in text_to_check or "fade out" in text_to_check or "[outro]" in text_to_check:
        fade_out_duration = 8.0  # 8 seconds professional fade-out
        
    if detected_key:
        logger.info(f"✨ 音楽のキー(調)の指定を検出しました: '{detected_key}'")
    else:
        logger.info("ℹ️ 明示的なキーの指定はありませんでした。AIの自動判定を使用します。")
        
    if detected_bpm:
        logger.info(f"✨ BPMの指定を検出しました: {detected_bpm}")
    if detected_ts:
        logger.info(f"✨ 拍子の指定を検出しました: '{detected_ts}'")
    if fade_out_duration > 0:
        logger.info(f"✨ アウトロ/フェードアウトの指定を検出しました。{fade_out_duration}秒の自動フェードアウト処理を適用します。")
    
    # 音楽パラメータのチューニング：ACE-Step公式アプリ（Gradio）のデフォルト値に近づけて自然なコード進行を保つ
    # negative_promptはLMに渡すと音楽コードを破壊して不協和音を生む原因になるため削除（デフォルトの "NO USER INPUT" に任せる）
    # ユーザーが明示的に長さを指定しない限り、AI(LM)の自律的な終了判断に任せる（Playgroundと同じ動作）
    actual_duration = args.duration if args.duration > 0 else 0
    
    if actual_duration == 0:
        logger.info("✨ 曲の長さ(Duration)はAIの自律判断(Auto)に委ねます（波状・ループを防止）")
    else:
        logger.info(f"✨ 指定された長さを使用します: {actual_duration}秒")


    # ---- 3.5. Enhance Caption — BYPASSED ----
    # エンハンスAI（ACE-Step内部LM）はバイパスします。
    # 理由: Ollamaが生成した短く的確なプロンプトに対し、エンハンスAIが毎回
    # 「synth pads」「lingering chord」等の持続音を誘発するキーワードを勝手に追加し、
    # SFTモデルのD#持続バグの主要原因となっていたため。
    # OllamaのプロンプトはアプローチE（30-50語制限）で既に最適化されているため、
    # エンハンスなしでPlaygroundと同等の品質が得られます。
    logger.info("✨ Ollamaが生成したキャプションをそのまま使用します（エンハンスAIはバイパス）")
    logger.info(f"✨ キャプション: {caption}")


    # ---- 3.6. Model Detection for Parameters ----
    model_id = getattr(dit_handler, 'model_id', '').lower()
    is_turbo = "turbo" in model_id
    is_sft = "sft" in model_id

    # SFTの場合は高解像度用パラメータ、Turboの場合は高速用パラメータを設定
    if is_turbo:
        inference_steps_val = 8
        shift_val = 1.0
        dcw_val = True
        guidance_scale_val = 7.0
        use_adg_val = False
        logger.info("⚡ Turboモデルを検出しました。Turbo用の設定を使用します (steps=8, shift=1.0, dcw=True)")
    elif is_sft:
        inference_steps_val = 50
        shift_val = 3.0
        dcw_val = False
        guidance_scale_val = 7.0  # Playgroundデフォルトに合わせる（Ollamaのプロンプトは既に短く制御済み）
        use_adg_val = False       # Playgroundデフォルト。ADGのangle_clipがコード進行の変化を鈍くする副作用を回避
        logger.info("✨ SFTモデルを検出しました。Playground準拠の設定を使用します (steps=50, shift=3.0, guidance=7.0, adg=False)")
    else:
        inference_steps_val = 32
        shift_val = 3.0
        dcw_val = False
        guidance_scale_val = 5.0
        use_adg_val = False
        logger.info("⚙️ Baseモデルを検出しました。Base用の設定を使用します (steps=32, shift=3.0, guidance=5.0)")

    params = GenerationParams(
        task_type="text2music",
        thinking=True,
        caption=caption,
        lyrics=lyrics_hiragana, # ひらがな歌詞をACE-Stepに渡して歌唱発音ミスを防ぐ
        vocal_language="ja",
        duration=actual_duration,
        keyscale=detected_key,
        bpm=detected_bpm,
        timesignature=detected_ts,
        fade_out_duration=fade_out_duration,
        inference_steps=inference_steps_val,
        guidance_scale=guidance_scale_val,
        use_adg=use_adg_val,
        sampler_mode="euler",# 安定・高速な標準サンプラーに戻す
        lm_temperature=0.8, # LMの温度を下げて、王道で破綻のないコード進行を生成させる
        shift=shift_val,
        dcw_enabled=dcw_val,
        seed=-1,
    )
    
    config = GenerationConfig(
        batch_size=1,
        audio_format=args.format,
        bit_depth=args.bit_depth,
    )
    
    # Save directly to a temp folder inside gradio_outputs
    temp_save_dir = os.path.join(str(PROJECT_ROOT), "gradio_outputs", "autochord_temp")
    os.makedirs(temp_save_dir, exist_ok=True)
    
    # Compose
    result = generate_music(
        dit_handler,
        llm_handler,
        params=params,
        config=config,
        save_dir=temp_save_dir,
    )
    
    # ---- 4. ファイルをユーザーのデスクトップへ移動 ----
    desktop_dir = os.path.expanduser("~/Desktop")
    if result.success and result.audios:
        logger.info("🎶 楽曲の生成が正常に完了しました！")
        for idx, audio in enumerate(result.audios):
            temp_path = audio.get("path")
            if temp_path and os.path.exists(temp_path):
                # 新しいファイル名を作成
                safe_theme = "".join([c for c in args.theme if c.isalnum() or c in " -_"])[:30]
                timestamp = int(time.time())
                final_name = f"Song_{safe_theme}_{timestamp}_{idx+1}.{args.format}"
                dest_path = os.path.join(desktop_dir, final_name)
                
                # デスクトップにコピー
                shutil.copy(temp_path, dest_path)
                logger.info(f"🎉 新しい楽曲をデスクトップに保存しました: {dest_path}")
                
                # Metadata (JSON) もコピーして、あとでSFTアップスケールに使えるようにする
                temp_json_path = os.path.splitext(temp_path)[0] + ".json"
                json_name = f"Song_{safe_theme}_{timestamp}_{idx+1}.json"
                json_dest_path = os.path.join(desktop_dir, json_name)
                if os.path.exists(temp_json_path):
                    shutil.copy(temp_json_path, json_dest_path)
                    logger.info(f"💾 メタデータ(JSON)をデスクトップに保存しました: {json_dest_path}")
                
                # 歌詞とキャプションのテキストファイルをデスクトップに出力
                lyrics_name = f"Lyrics_{safe_theme}_{timestamp}_{idx+1}.txt"
                lyrics_path = os.path.join(desktop_dir, lyrics_name)
                try:
                    with open(lyrics_path, "w", encoding="utf-8") as f:
                        f.write("=====================================================\n")
                        f.write(f"🎵 Auto-Composer: Generated Lyrics & Caption\n")
                        f.write(f"   Theme: {args.theme}\n")
                        if actual_duration > 0:
                            f.write(f"   Target Duration: {actual_duration}s\n")
                        f.write("=====================================================\n\n")
                        f.write("■ 歌詞（漢字かな混じり - 通常表示用）\n")
                        f.write("-----------------------------------------------------\n")
                        f.write(lyrics)
                        f.write("\n\n")
                        f.write("■ 歌詞（ひらがな - ACE-Step 歌唱入力用）\n")
                        f.write("-----------------------------------------------------\n")
                        f.write(lyrics_hiragana)
                        f.write("\n\n")
                        f.write("■ 音楽構成プロンプト (Music Caption - Ollama Base)\n")
                        f.write("-----------------------------------------------------\n")
                        f.write(original_caption if 'original_caption' in locals() else caption)
                        f.write("\n\n")
                        f.write("■ ✨ エンハンス済みプロンプト (Enhanced Caption - ACE-Step LM)\n")
                        f.write("-----------------------------------------------------\n")
                        f.write(caption)
                        f.write("\n")
                    logger.info(f"📝 歌詞のテキストファイルをデスクトップに保存しました: {lyrics_path}")
                except Exception as lyrics_err:
                    logger.warning(f"歌詞テキストファイルの保存に失敗しました: {str(lyrics_err)}")
                
                # 自動でファイルを開いてユーザーに通知
                try:
                    import platform
                    if platform.system() == "Darwin":
                        # QuickTime Playerを最前面に表示し、自動再生するAppleScript
                        play_script = (
                            f"osascript "
                            f"-e 'tell application \"QuickTime Player\" to activate' "
                            f"-e 'tell application \"QuickTime Player\" to open POSIX file \"{dest_path}\"' "
                            f"-e 'tell application \"QuickTime Player\" to play document 1'"
                        )
                        os.system(play_script)
                        if 'lyrics_path' in locals() and os.path.exists(lyrics_path):
                            os.system(f"open '{lyrics_path}'")
                except Exception:
                    pass
                
        # クリーニング
        shutil.rmtree(temp_save_dir, ignore_errors=True)
    else:
        logger.error(f"楽曲の生成に失敗しました: {result.status_message}")
        shutil.rmtree(temp_save_dir, ignore_errors=True)
        sys.exit(1)
        
    logger.info(f"✅ 全ての処理が完了しました。かかった時間: {time.time() - t_start:.1f} 秒")

if __name__ == "__main__":
    main()
