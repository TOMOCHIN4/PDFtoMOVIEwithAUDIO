"""
PDFtoMOVIEwithAUDIO - Hugging Face Space Application
PDFをナレーション付き動画に変換
"""

import gradio as gr
from google import genai
from google.genai import types
import os
import tempfile
import wave
import time
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from pydub import AudioSegment
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip
import fitz  # PyMuPDF
from huggingface_hub import HfApi
import datetime
import json
import re
import traceback
from pydantic import BaseModel
from typing import List


# ===========================
# 構造化出力スキーマ
# ===========================
class DialogueLine(BaseModel):
    """2人モードの1セリフ"""
    speaker: str
    text: str


class PageScriptSingle(BaseModel):
    """1人モード: 1ページ分のナレーション"""
    page_number: int
    narration: str


class PageScriptMulti(BaseModel):
    """2人モード: 1ページ分の対話"""
    page_number: int
    dialogue: List[DialogueLine]


class ScriptResponseSingle(BaseModel):
    """1人モード: 全ページのナレーション"""
    pages: List[PageScriptSingle]


class ScriptResponseMulti(BaseModel):
    """2人モード: 全ページの対話"""
    pages: List[PageScriptMulti]


# ===========================
# リトライ機構
# ===========================
def call_with_retry(func, *args, max_retries=3, initial_delay=60, **kwargs):
    """レートリミット対応のリトライ機構"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = initial_delay * (attempt + 1)  # 60s, 120s, 180s
                print(f"[retry] レートリミット検出。{delay}秒待機後にリトライ ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise  # 429以外のエラーはそのまま再送出
    # 最終リトライ
    return func(*args, **kwargs)


# ===========================
# 設定
# ===========================
PAGES_PER_CHUNK = 5
AUDIO_SPEED = 1.2
SILENCE_BEFORE = 1000
SILENCE_AFTER = 500
OUTPUT_FPS = 24
OUTPUT_RESOLUTION = (1280, 720)  # HD画質（高速化）

# 環境変数
ENV_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENV_HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_HF_REPO_ID = os.environ.get("HF_REPO_ID", "leave-everything/PDFtoMOVIEwithAUDIO")

# 番組スタイル
PROGRAM_STYLES = {
    "1人ラジオ風": {
        "speakers": 1,
        "speaker_config": {"host": {"name": "ホスト", "voice": "Kore"}},
        "script_prompt": "親しみやすいラジオDJとして、リスナーに語りかけるような温かみのある口調でPDFの内容を解説してください。",
        "tts_style": "親しみやすいラジオDJのように、温かく語りかけるように読み上げてください。"
    },
    "2人ポッドキャスト風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "タケシ", "voice": "Kore"},
            "guest": {"name": "ユミ", "voice": "Puck"}
        },
        "script_prompt": "ホスト「タケシ」とアシスタント「ユミ」の掛け合いでPDFの内容を解説してください。タケシは落ち着いた解説役、ユミは質問したり感想を述べる役割です。",
        "tts_style_host": "落ち着いた男性ポッドキャスターとして読み上げてください。",
        "tts_style_guest": "明るく好奇心旺盛な女性アシスタントとして話してください。"
    },
    "2人漫才風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "ツッコミ", "voice": "Charon"},
            "guest": {"name": "ボケ", "voice": "Fenrir"}
        },
        "script_prompt": "「ツッコミ」と「ボケ」の掛け合いでPDFの内容を面白おかしく解説してください。ボケが内容を誤解し、ツッコミが正しく訂正する形式です。",
        "tts_style_host": "漫才のツッコミ役として、テンポよくキレのあるツッコミを入れてください。",
        "tts_style_guest": "漫才のボケ役として、少しとぼけた感じで話してください。"
    },
    "1人ニュース風": {
        "speakers": 1,
        "speaker_config": {"host": {"name": "キャスター", "voice": "Alnilam"}},
        "script_prompt": "ニュースキャスターとして、客観的かつ明確にPDFの内容を伝えてください。",
        "tts_style": "プロのニュースキャスターとして、明瞭で落ち着いた口調で読み上げてください。"
    },
    "1人講義風": {
        "speakers": 1,
        "speaker_config": {"host": {"name": "教授", "voice": "Charon"}},
        "script_prompt": "大学教授として、講義形式でPDFの内容をわかりやすく解説してください。",
        "tts_style": "知識豊富な大学教授として、落ち着いて丁寧に解説してください。"
    },
    "2人インタビュー風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "インタビュアー", "voice": "Aoede"},
            "guest": {"name": "専門家", "voice": "Charon"}
        },
        "script_prompt": "「インタビュアー」が「専門家」にPDFの内容について質問し、専門家が詳しく回答する形式です。",
        "tts_style_host": "好奇心旺盛なインタビュアーとして質問してください。",
        "tts_style_guest": "専門家として自信を持って丁寧に説明してください。"
    }
}


def split_pdf(pdf_path, pages_per_chunk=5):
    """PDFを指定ページ数ごとに分割"""
    print(f"[split_pdf] PDF分割開始")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[split_pdf] 総ページ数: {total_pages}, {pages_per_chunk}ページごとに分割")
    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        chunk_doc = fitz.open()

        for page_num in range(start, end):
            chunk_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        chunk_path = tempfile.mktemp(suffix='.pdf')
        chunk_doc.save(chunk_path)
        chunk_doc.close()

        page_numbers = list(range(start + 1, end + 1))
        chunks.append((chunk_path, page_numbers))

    doc.close()
    print(f"[split_pdf] 分割完了: {len(chunks)}チャンク作成")
    return chunks


def pdf_to_images(pdf_path, dpi=150):
    """PDFを画像に変換"""
    print(f"[pdf_to_images] 画像変換開始")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"[pdf_to_images] 変換完了: {len(images)}ページ")
    return images


def generate_narration_script(pdf_chunk_path, page_numbers, program_style, api_key, chunk_index, total_chunks, total_pages):
    """Gemini APIでナレーション台本を生成（構造化出力）"""
    print(f"[generate_script] 台本生成開始: ページ {page_numbers} (チャンク {chunk_index}/{total_chunks})")
    client = genai.Client(api_key=api_key)

    with open(pdf_chunk_path, 'rb') as f:
        pdf_data = f.read()

    speaker_info = program_style["speaker_config"]
    speaker_names = [info["name"] for info in speaker_info.values()]
    is_single_speaker = program_style["speakers"] == 1

    # 構造化出力スキーマを選択
    if is_single_speaker:
        response_schema = ScriptResponseSingle
        format_instruction = f"""
出力形式: 以下のJSON構造で出力してください。
- pages: 各ページのナレーションを含むリスト
  - page_number: ページ番号（{page_numbers[0]}から{page_numbers[-1]}）
  - narration: そのページのナレーション全文（15〜30秒で読める長さ）
"""
    else:
        response_schema = ScriptResponseMulti
        format_instruction = f"""
出力形式: 以下のJSON構造で出力してください。
- pages: 各ページの対話を含むリスト
  - page_number: ページ番号（{page_numbers[0]}から{page_numbers[-1]}）
  - dialogue: 話者と発言のリスト
    - speaker: 話者名（「{speaker_names[0]}」または「{speaker_names[1]}」）
    - text: セリフ

各ページにつき2〜4往復の対話を含めてください。
"""

    # チャンク位置に応じた構成指示
    if total_chunks == 1:
        position_instruction = "これは単独の資料です。適切な導入と締めくくりを含めてください。"
    elif chunk_index == 1:
        position_instruction = f"""これはPDF全体（{total_pages}ページ）の冒頭部分（{chunk_index}/{total_chunks}チャンク目）です。
- 番組の導入として、これから何を解説するかを簡潔に紹介してください
- ただし「締めくくり」や「まとめ」は不要です（続きがあるため）"""
    elif chunk_index == total_chunks:
        if total_chunks == 2:
            position_instruction = f"""これはPDF全体（{total_pages}ページ）の後半部分（2/2チャンク目＝最終）です。
- 冒頭の導入は不要です（前半で既に導入済み）
- 最後のページで全体のまとめや締めくくりを入れてください"""
        else:
            position_instruction = f"""これはPDF全体（{total_pages}ページ）の最終部分（{chunk_index}/{total_chunks}チャンク目）です。
- 冒頭の導入は不要です（すでに番組は始まっています）
- 最後のページで全体のまとめや締めくくりを入れてください"""
    else:
        position_instruction = f"""これはPDF全体（{total_pages}ページ）の中盤部分（{chunk_index}/{total_chunks}チャンク目）です。
- 冒頭の導入は不要です（すでに番組は始まっています）
- 締めくくりも不要です（まだ続きがあります）
- 内容の解説に集中してください"""

    prompt = f"""
あなたは優秀なナレーション台本ライターです。

【重要：番組の構成位置】
{position_instruction}

【重要な指示】
1. 添付されたPDFの各ページを注意深く読み取ってください
2. テキスト、図表、画像、グラフなど全ての要素を認識してください
3. 各ページの内容を正確に理解した上で、ナレーション台本を作成してください
4. 必ず指定されたすべてのページ（{len(page_numbers)}ページ分）の台本を生成してください

【台本作成ルール】
{program_style["script_prompt"]}

【形式要件】
- 各ページ15〜30秒程度で読める長さにしてください（簡潔に）
- PDFの実際の内容に基づいて、具体的で情報豊富なナレーションを作成してください
- 「ページXの内容です」のような曖昧な表現は禁止です
- 必ずPDFに書かれている具体的な情報、データ、説明を盛り込んでください

{format_instruction}

対象ページ番号: {page_numbers}

PDFの内容を詳細に分析し、視聴者にとって価値のあるナレーション台本を作成してください。
"""

    # 構造化出力でAPI呼び出し
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )
    )

    # 構造化されたレスポンスをパース
    try:
        response_data = json.loads(response.text)
        print(f"[generate_script] 構造化出力成功: {len(response_data.get('pages', []))}ページ分")
    except json.JSONDecodeError as e:
        print(f"[generate_script] JSONパースエラー: {e}")
        print(f"[generate_script] レスポンス: {response.text[:500]}")
        response_data = {"pages": []}

    # 結果を変換（page_number → 実際のページ番号でマッピング）
    result = {}
    pages_data = response_data.get("pages", [])

    for page_info in pages_data:
        page_num = page_info.get("page_number")
        if page_num in page_numbers:
            if is_single_speaker:
                result[page_num] = page_info.get("narration", f"ページ{page_num}の内容です。")
            else:
                dialogue = page_info.get("dialogue", [])
                result[page_num] = [{"speaker": d.get("speaker", speaker_names[0]), "text": d.get("text", "")} for d in dialogue]

    # 欠落ページのフォールバック
    for page_num in page_numbers:
        if page_num not in result:
            print(f"[generate_script] 警告: ページ{page_num}の台本が欠落、フォールバック使用")
            if is_single_speaker:
                result[page_num] = f"ページ{page_num}の内容について説明します。"
            else:
                result[page_num] = [
                    {"speaker": speaker_names[0], "text": f"ページ{page_num}について見ていきましょう。"},
                    {"speaker": speaker_names[1], "text": "はい、お願いします。"}
                ]

    print(f"[generate_script] 台本生成完了: {len(result)}ページ分")
    return result


def text_to_speech_single(text, voice_name, style_prompt, api_key):
    """1人用TTS（レートリミット対応）"""
    print(f"[TTS] 音声生成開始 (1人モード, voice={voice_name})")
    client = genai.Client(api_key=api_key)

    full_prompt = f"""{style_prompt}

【重要な読み上げ指示】
- やや早口でテンポよく読み上げてください
- 日本語の発音は正確に、滑舌よくはっきりと発声してください
- 聞き取りやすさを維持しながらスピード感のある読み上げをしてください

以下のテキストを読み上げてください:
{text}"""

    def _call_tts():
        return client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            )
        )

    response = call_with_retry(_call_tts)
    print(f"[TTS] 音声生成完了 (1人モード)")
    return response.candidates[0].content.parts[0].inline_data.data


def text_to_speech_multi(dialogue, speaker_config, style_prompts, api_key):
    """2人用マルチスピーカーTTS（レートリミット対応）"""
    print(f"[TTS] 音声生成開始 (2人モード, {len(dialogue)}セリフ)")
    client = genai.Client(api_key=api_key)

    conversation_text = ""
    for line in dialogue:
        conversation_text += f"{line['speaker']}: {line['text']}\n"

    host_info = speaker_config["host"]
    guest_info = speaker_config["guest"]

    style_instruction = f"""
以下の会話を2人の話者で読み上げてください。

{host_info["name"]}の話し方: {style_prompts.get("host", "自然に話してください")}
{guest_info["name"]}の話し方: {style_prompts.get("guest", "自然に話してください")}

【重要な読み上げ指示】
- やや早口でテンポよく読み上げてください
- 日本語の発音は正確に、滑舌よくはっきりと発声してください
- 掛け合いのテンポ感を大切に、スピード感のある会話にしてください

会話:
{conversation_text}
"""

    def _call_tts():
        return client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=style_instruction,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker=host_info["name"],
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=host_info["voice"],
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker=guest_info["name"],
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=guest_info["voice"],
                                    )
                                )
                            ),
                        ]
                    )
                ),
            )
        )

    response = call_with_retry(_call_tts)
    print(f"[TTS] 音声生成完了 (2人モード)")
    return response.candidates[0].content.parts[0].inline_data.data


def save_pcm_to_wav(pcm_data, output_path, sample_rate=24000, channels=1, sample_width=2):
    """PCMをWAVに保存"""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def process_audio(wav_path, speed=1.2, silence_before_ms=1000, silence_after_ms=500):
    """音声処理: 速度変換、無音追加"""
    print(f"[process_audio] 音声処理開始 (速度={speed}x)")
    audio = AudioSegment.from_wav(wav_path)

    new_sample_rate = int(audio.frame_rate * speed)
    speed_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(audio.frame_rate)

    silence_before = AudioSegment.silent(duration=silence_before_ms)
    silence_after = AudioSegment.silent(duration=silence_after_ms)

    final_audio = silence_before + speed_audio + silence_after

    output_path = tempfile.mktemp(suffix='.wav')
    final_audio.export(output_path, format='wav')

    duration = len(final_audio) / 1000.0
    print(f"[process_audio] 音声処理完了 (長さ={duration:.1f}秒)")

    return output_path, duration


def resize_image_for_video(image, target_size=(1920, 1080)):
    """画像を動画用にリサイズ"""
    target_w, target_h = target_size

    img_ratio = image.width / image.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    result = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    result.paste(resized, offset)

    return result


def create_page_video(image, audio_path, duration):
    """ページ動画を作成"""
    print(f"[create_video] ページ動画作成開始 (長さ={duration:.1f}秒)")
    resized_img = resize_image_for_video(image, OUTPUT_RESOLUTION)

    img_path = tempfile.mktemp(suffix='.png')
    resized_img.save(img_path)

    img_clip = ImageClip(img_path, duration=duration)
    audio_clip = AudioFileClip(audio_path)

    video = img_clip.with_audio(audio_clip)

    output_path = tempfile.mktemp(suffix='.mp4')
    video.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec='libx264',
        audio_codec='aac',
        logger="bar"
    )

    img_clip.close()
    audio_clip.close()
    os.remove(img_path)
    print(f"[create_video] ページ動画作成完了")

    return output_path


def merge_videos(video_paths, output_path):
    """動画を結合（ffmpeg直接結合で高速化）"""
    import subprocess

    # ファイルリストを作成
    list_path = tempfile.mktemp(suffix='.txt')
    with open(list_path, 'w') as f:
        for path in video_paths:
            # ffmpeg concat demuxer形式
            f.write(f"file '{path}'\n")

    # ffmpegで再エンコードなしに結合（超高速）
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_path,
        '-c', 'copy',  # 再エンコードなし
        output_path
    ]

    print(f"[merge_videos] ffmpeg直接結合開始: {len(video_paths)}本の動画")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[merge_videos] ffmpeg警告/エラー: {result.stderr}")
        # フォールバック: moviepyで結合
        print("[merge_videos] フォールバック: moviepyで結合")
        clips = [VideoFileClip(path) for path in video_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            fps=OUTPUT_FPS,
            codec='libx264',
            audio_codec='aac',
            logger="bar"
        )
        for clip in clips:
            clip.close()
        final.close()
    else:
        print(f"[merge_videos] ffmpeg結合完了")

    # クリーンアップ
    os.remove(list_path)


def upload_to_hf_dataset(video_path, hf_token, repo_id):
    """HFにアップロード"""
    print(f"[upload] HFアップロード開始: {repo_id}")
    api = HfApi()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pdf_movie_{timestamp}.mp4"

    url = api.upload_file(
        path_or_fileobj=video_path,
        path_in_repo=f"videos/{filename}",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )

    print(f"[upload] アップロード完了: {url}")
    return url


def process_pdf_to_movie(pdf_file, program_style_name, gemini_api_key, hf_token, hf_repo_id, progress=gr.Progress()):
    """メイン処理"""
    print(f"=" * 50)
    print(f"[main] PDF→動画変換開始")
    print(f"[main] スタイル: {program_style_name}")
    print(f"=" * 50)

    if pdf_file is None:
        return None, "PDFファイルをアップロードしてください", ""

    api_key = gemini_api_key or ENV_GEMINI_API_KEY
    token = hf_token or ENV_HF_TOKEN
    repo_id = hf_repo_id or ENV_HF_REPO_ID

    if not api_key:
        return None, "Gemini APIキーを入力してください", ""

    if not token or not repo_id:
        return None, "HFトークンとリポジトリIDを入力してください", ""

    try:
        pdf_path = pdf_file
        program_style = PROGRAM_STYLES.get(program_style_name, PROGRAM_STYLES["1人ラジオ風"])

        progress(0.05, desc="PDFを分割中...")
        chunks = split_pdf(pdf_path, PAGES_PER_CHUNK)
        total_pages = sum(len(pages) for _, pages in chunks)

        progress(0.1, desc=f"PDF分割完了: {total_pages}ページ")

        all_images = pdf_to_images(pdf_path)

        all_scripts = {}
        total_chunks = len(chunks)
        for i, (chunk_path, page_numbers) in enumerate(chunks):
            chunk_index = i + 1  # 1-indexed
            progress(0.1 + (0.3 * i / len(chunks)),
                    desc=f"台本生成中... {chunk_index}/{total_chunks}")

            scripts = generate_narration_script(
                chunk_path, page_numbers, program_style, api_key,
                chunk_index=chunk_index, total_chunks=total_chunks, total_pages=total_pages
            )
            all_scripts.update(scripts)
            os.remove(chunk_path)

        progress(0.4, desc="音声生成中...")

        page_data = []
        for i, page_num in enumerate(range(1, total_pages + 1)):
            progress(0.4 + (0.4 * i / total_pages),
                    desc=f"音声生成中... {page_num}/{total_pages}")

            script = all_scripts.get(page_num)

            if program_style["speakers"] == 1:
                narration = script if isinstance(script, str) else f"ページ{page_num}です。"
                host_config = program_style["speaker_config"]["host"]

                pcm_data = text_to_speech_single(
                    narration,
                    host_config["voice"],
                    program_style.get("tts_style", "自然に読み上げてください。"),
                    api_key
                )
            else:
                dialogue = script if isinstance(script, list) else [
                    {"speaker": program_style["speaker_config"]["host"]["name"],
                     "text": f"ページ{page_num}について見ていきましょう。"}
                ]

                style_prompts = {
                    "host": program_style.get("tts_style_host", "自然に話してください。"),
                    "guest": program_style.get("tts_style_guest", "自然に話してください。")
                }

                pcm_data = text_to_speech_multi(
                    dialogue,
                    program_style["speaker_config"],
                    style_prompts,
                    api_key
                )

            wav_path = tempfile.mktemp(suffix='.wav')
            save_pcm_to_wav(pcm_data, wav_path)

            processed_path, duration = process_audio(wav_path, AUDIO_SPEED, SILENCE_BEFORE, SILENCE_AFTER)
            os.remove(wav_path)

            page_data.append((all_images[page_num - 1], processed_path, duration))

        progress(0.8, desc="動画作成中...")

        video_paths = []
        for i, (image, audio_path, duration) in enumerate(page_data):
            progress(0.8 + (0.15 * i / len(page_data)),
                    desc=f"動画作成中... {i+1}/{len(page_data)}")

            video_path = create_page_video(image, audio_path, duration)
            video_paths.append(video_path)
            os.remove(audio_path)

        progress(0.95, desc="動画結合中...")

        final_video_path = tempfile.mktemp(suffix='.mp4')
        merge_videos(video_paths, final_video_path)

        for path in video_paths:
            os.remove(path)

        progress(0.98, desc="HFにアップロード中...")

        hf_url = upload_to_hf_dataset(final_video_path, token, repo_id)

        progress(1.0, desc="完了!")

        print(f"=" * 50)
        print(f"[main] 処理完了!")
        print(f"[main] 総ページ数: {total_pages}")
        print(f"[main] 保存先: {hf_url}")
        print(f"=" * 50)

        status_msg = f"""
完了!

処理情報:
- 総ページ数: {total_pages}
- 番組スタイル: {program_style_name}
- 話者数: {program_style["speakers"]}人

保存先: {hf_url}
"""

        return final_video_path, status_msg, hf_url

    except Exception as e:
        print(f"[main] エラー発生: {str(e)}")
        print(traceback.format_exc())
        error_msg = f"エラー: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, ""


# ===========================
# Gradio UI (シンプル版)
# ===========================

def create_demo():
    with gr.Blocks(title="PDFtoMOVIEwithAUDIO") as demo:
        gr.Markdown("# PDFtoMOVIEwithAUDIO")
        gr.Markdown("PDFをナレーション付き動画に自動変換します。")

        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="PDFファイル", file_types=[".pdf"])

                program_style = gr.Dropdown(
                    choices=list(PROGRAM_STYLES.keys()),
                    value="1人ラジオ風",
                    label="番組スタイル"
                )

                gemini_key = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="環境変数設定済みなら空欄可"
                )

                hf_token = gr.Textbox(
                    label="HF Token",
                    type="password",
                    placeholder="環境変数設定済みなら空欄可"
                )

                hf_repo = gr.Textbox(
                    label="HF Dataset Repo ID",
                    value=ENV_HF_REPO_ID
                )

                generate_btn = gr.Button("動画生成", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="生成動画")
                status_output = gr.Textbox(label="ステータス", lines=10)
                hf_url_output = gr.Textbox(label="HF URL")

        generate_btn.click(
            fn=process_pdf_to_movie,
            inputs=[pdf_input, program_style, gemini_key, hf_token, hf_repo],
            outputs=[video_output, status_output, hf_url_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
