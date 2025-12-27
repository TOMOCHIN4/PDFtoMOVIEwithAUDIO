"""
PDFtoMOVIEwithAUDIO - Hugging Face Space Application
PDFを動画に変換するシステム（ナレーション音声付き）
"""

import gradio as gr
from google import genai
from google.genai import types
import os
import tempfile
import wave
import base64
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from pydub import AudioSegment
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip
import fitz  # PyMuPDF
from huggingface_hub import HfApi, upload_file
import datetime
import json
import io
import re


# ===========================
# 設定
# ===========================
PAGES_PER_CHUNK = 5  # PDF分割単位
AUDIO_SPEED = 1.2    # 音声速度倍率
SILENCE_BEFORE = 1000  # 前の無音（ミリ秒）
SILENCE_AFTER = 500    # 後の無音（ミリ秒）
OUTPUT_FPS = 24
OUTPUT_RESOLUTION = (1920, 1080)

# 環境変数からAPIキーを取得（HF Spacesのシークレット対応）
ENV_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENV_HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_HF_REPO_ID = os.environ.get("HF_REPO_ID", "leave-everything/PDFtoMOVIEwithAUDIO")

# 番組スタイルプリセット
PROGRAM_STYLES = {
    "1人ラジオ風": {
        "speakers": 1,
        "speaker_config": {
            "host": {"name": "ホスト", "voice": "Kore"}
        },
        "script_prompt": """
あなたは親しみやすいラジオDJです。
リスナーに語りかけるような温かみのある口調で、PDFの内容を解説してください。
「皆さん、こんにちは」「さて、次は...」「いかがでしたか？」などのラジオ的な言い回しを適度に入れてください。
一人称は「私」、リスナーへの呼びかけは「皆さん」「リスナーの皆さん」を使ってください。
""",
        "tts_style": "親しみやすいラジオDJのように、温かく語りかけるように読み上げてください。"
    },
    "2人ポッドキャスト風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "タケシ", "voice": "Kore"},
            "guest": {"name": "ユミ", "voice": "Puck"}
        },
        "script_prompt": """
あなたは2人組のポッドキャスターです。
ホスト「タケシ」とアシスタント「ユミ」の掛け合いでPDFの内容を解説してください。
タケシは落ち着いた解説役、ユミは質問したり感想を述べたりする役割です。

会話形式で出力してください：
タケシ: （セリフ）
ユミ: （セリフ）

自然な会話のキャッチボールを心がけ、「なるほど〜」「それって〇〇ってことですか？」「面白いですね！」などの相槌も入れてください。
""",
        "tts_style_host": "落ち着いた男性ポッドキャスターとして、わかりやすく解説するように読み上げてください。",
        "tts_style_guest": "明るく好奇心旺盛な女性アシスタントとして、楽しそうに話してください。"
    },
    "2人漫才風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "ツッコミ", "voice": "Charon"},
            "guest": {"name": "ボケ", "voice": "Fenrir"}
        },
        "script_prompt": """
あなたは漫才コンビです。
「ツッコミ」と「ボケ」の掛け合いでPDFの内容を面白おかしく解説してください。
ボケが内容を誤解したり大げさに解釈したりして、ツッコミが正しく訂正する形式です。

会話形式で出力してください：
ツッコミ: （セリフ）
ボケ: （セリフ）

「なんでやねん！」「ちゃうちゃう」「そうそう、それそれ」などの漫才的なやり取りを入れつつ、
最終的には正しい情報が伝わるようにしてください。テンポよく、笑いも交えて！
""",
        "tts_style_host": "漫才のツッコミ役として、テンポよくキレのあるツッコミを入れてください。",
        "tts_style_guest": "漫才のボケ役として、少しとぼけた感じで、大げさなリアクションをしてください。"
    },
    "1人ニュース風": {
        "speakers": 1,
        "speaker_config": {
            "host": {"name": "キャスター", "voice": "Alnilam"}
        },
        "script_prompt": """
あなたはニュースキャスターです。
報道番組のように、客観的かつ明確にPDFの内容を伝えてください。
「本日お伝えするのは...」「続いては...」「以上、〇〇についてお伝えしました」などのニュース的な言い回しを使ってください。
敬体（です・ます調）で、正確で簡潔な表現を心がけてください。
""",
        "tts_style": "プロのニュースキャスターとして、明瞭で落ち着いた口調で読み上げてください。"
    },
    "1人講義風": {
        "speakers": 1,
        "speaker_config": {
            "host": {"name": "教授", "voice": "Charon"}
        },
        "script_prompt": """
あなたは大学教授です。
講義形式でPDFの内容をわかりやすく解説してください。
「今日のテーマは...」「ここで重要なのは...」「つまり...」「例えば...」などの教育的な言い回しを使ってください。
専門用語は噛み砕いて説明し、聴講者が理解しやすいように心がけてください。
""",
        "tts_style": "知識豊富な大学教授として、落ち着いて丁寧に、わかりやすく解説してください。"
    },
    "2人インタビュー風": {
        "speakers": 2,
        "speaker_config": {
            "host": {"name": "インタビュアー", "voice": "Aoede"},
            "guest": {"name": "専門家", "voice": "Charon"}
        },
        "script_prompt": """
あなたはインタビュー番組の出演者です。
「インタビュアー」が「専門家」にPDFの内容について質問し、専門家が詳しく回答する形式です。

会話形式で出力してください：
インタビュアー: （セリフ）
専門家: （セリフ）

インタビュアーは視聴者目線で素朴な疑問を投げかけ、専門家は丁寧かつ専門的に回答してください。
「〇〇について教えていただけますか？」「それは興味深いですね」などの自然なやり取りを入れてください。
""",
        "tts_style_host": "好奇心旺盛なインタビュアーとして、興味を持って質問してください。",
        "tts_style_guest": "その分野の専門家として、自信を持って丁寧に説明してください。"
    }
}


def split_pdf(pdf_path: str, pages_per_chunk: int = 5) -> list:
    """
    PDFを指定ページ数ごとに分割
    Returns: [(chunk_path, page_numbers), ...]
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
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
    return chunks


def pdf_to_images(pdf_path: str, dpi: int = 150) -> list:
    """PDFを画像に変換"""
    images = convert_from_path(pdf_path, dpi=dpi)
    return images


def generate_narration_script(pdf_chunk_path: str, page_numbers: list, 
                              program_style: dict, api_key: str) -> dict:
    """
    Gemini APIを使用してPDFからナレーション台本を生成
    Returns: {page_number: {"script": text, "speakers": [...]}, ...}
    """
    client = genai.Client(api_key=api_key)
    
    # PDFを読み込んでbase64エンコード
    with open(pdf_chunk_path, 'rb') as f:
        pdf_data = f.read()
    
    # スピーカー情報を構築
    speaker_info = program_style["speaker_config"]
    speaker_names = [info["name"] for info in speaker_info.values()]
    
    if program_style["speakers"] == 1:
        format_instruction = """
出力形式（厳守してください）:
各ページのナレーションをJSON形式で出力してください。
```json
{
    "page_1": "ここにページ1のナレーション全文...",
    "page_2": "ここにページ2のナレーション全文...",
    ...
}
```
"""
    else:
        format_instruction = f"""
出力形式（厳守してください）:
各ページの会話をJSON形式で出力してください。話者名は必ず「{speaker_names[0]}」「{speaker_names[1]}」を使用してください。

```json
{{
    "page_1": [
        {{"speaker": "{speaker_names[0]}", "text": "セリフ1"}},
        {{"speaker": "{speaker_names[1]}", "text": "セリフ2"}},
        ...
    ],
    "page_2": [
        {{"speaker": "{speaker_names[0]}", "text": "セリフ1"}},
        ...
    ],
    ...
}}
```
"""
    
    prompt = f"""
{program_style["script_prompt"]}

以下のPDFの各ページについて、上記のスタイルでナレーション台本を作成してください。

要件:
1. 各ページの内容を分かりやすく説明する
2. 各ページ30秒〜1分程度で読める長さ
3. 箇条書きや図表がある場合は、その内容を口頭で説明する
4. 番組の流れとして自然になるよう、ページ間のつなぎも意識する

{format_instruction}

対象ページ番号: {page_numbers}
"""
    
    response = client.models.generate_content(
        model="gemini-3.0-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(
                        data=pdf_data,
                        mime_type="application/pdf"
                    ),
                    types.Part.from_text(prompt)
                ]
            )
        ]
    )
    
    # JSONを抽出
    response_text = response.text
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text
    
    try:
        scripts = json.loads(json_str)
    except json.JSONDecodeError:
        # フォールバック
        scripts = {}
        for i, page_num in enumerate(page_numbers):
            if program_style["speakers"] == 1:
                scripts[f"page_{i+1}"] = f"ページ{page_num}の内容を説明します。"
            else:
                scripts[f"page_{i+1}"] = [
                    {"speaker": speaker_names[0], "text": f"ページ{page_num}について見ていきましょう。"},
                    {"speaker": speaker_names[1], "text": "はい、お願いします。"}
                ]
    
    # ページ番号を実際の番号にマッピング
    result = {}
    for i, page_num in enumerate(page_numbers):
        key = f"page_{i+1}"
        if key in scripts:
            result[page_num] = scripts[key]
        else:
            if program_style["speakers"] == 1:
                result[page_num] = f"ページ{page_num}の内容です。"
            else:
                result[page_num] = [
                    {"speaker": speaker_names[0], "text": f"ページ{page_num}の内容です。"},
                    {"speaker": speaker_names[1], "text": "なるほど。"}
                ]
    
    return result


def text_to_speech_single(text: str, voice_name: str, style_prompt: str, 
                          api_key: str) -> bytes:
    """
    Gemini TTS APIを使用してテキストを音声に変換（1人用）
    Returns: PCM audio data
    """
    client = genai.Client(api_key=api_key)
    
    # スタイル付きプロンプト
    full_prompt = f"{style_prompt}\n\n以下のテキストを読み上げてください:\n{text}"
    
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-tts",
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
    
    # 音声データを取得
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    return audio_data


def text_to_speech_multi(dialogue: list, speaker_config: dict, 
                         style_prompts: dict, api_key: str) -> bytes:
    """
    Gemini TTS APIを使用して会話を音声に変換（2人用マルチスピーカー）
    dialogue: [{"speaker": "名前", "text": "セリフ"}, ...]
    speaker_config: {"host": {"name": "...", "voice": "..."}, "guest": {...}}
    style_prompts: {"host": "スタイル", "guest": "スタイル"}
    Returns: PCM audio data
    """
    client = genai.Client(api_key=api_key)
    
    # スピーカー名からロール（host/guest）へのマッピングを作成
    name_to_role = {}
    for role, info in speaker_config.items():
        name_to_role[info["name"]] = role
    
    # 会話テキストを構築
    conversation_text = ""
    for line in dialogue:
        speaker = line["speaker"]
        text = line["text"]
        conversation_text += f"{speaker}: {text}\n"
    
    # スタイルプロンプトを構築
    host_info = speaker_config["host"]
    guest_info = speaker_config["guest"]
    
    style_instruction = f"""
以下の会話を2人の話者で読み上げてください。

{host_info["name"]}の話し方: {style_prompts.get("host", "自然に話してください")}
{guest_info["name"]}の話し方: {style_prompts.get("guest", "自然に話してください")}

会話:
{conversation_text}
"""
    
    # マルチスピーカーTTS
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-tts",
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
    
    # 音声データを取得
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    return audio_data


def save_pcm_to_wav(pcm_data: bytes, output_path: str, 
                    sample_rate: int = 24000, channels: int = 1, 
                    sample_width: int = 2):
    """PCMデータをWAVファイルとして保存"""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def process_audio(wav_path: str, speed: float = 1.2, 
                  silence_before_ms: int = 1000, 
                  silence_after_ms: int = 500) -> tuple:
    """
    音声を処理: 速度変換、無音追加
    Returns: (processed_wav_path, duration_seconds)
    """
    audio = AudioSegment.from_wav(wav_path)
    
    # 速度変換（ピッチを維持しつつ速度を変更）
    # pydubでの速度変更はサンプルレート変更で実現
    new_sample_rate = int(audio.frame_rate * speed)
    speed_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(audio.frame_rate)
    
    # 無音を追加
    silence_before = AudioSegment.silent(duration=silence_before_ms)
    silence_after = AudioSegment.silent(duration=silence_after_ms)
    
    final_audio = silence_before + speed_audio + silence_after
    
    # 保存
    output_path = tempfile.mktemp(suffix='.wav')
    final_audio.export(output_path, format='wav')
    
    duration = len(final_audio) / 1000.0  # ミリ秒を秒に変換
    
    return output_path, duration


def resize_image_for_video(image: Image.Image, 
                           target_size: tuple = (1920, 1080)) -> Image.Image:
    """画像を動画用にリサイズ（アスペクト比維持、余白は黒）"""
    target_w, target_h = target_size
    
    # アスペクト比を計算
    img_ratio = image.width / image.height
    target_ratio = target_w / target_h
    
    if img_ratio > target_ratio:
        # 横長 - 幅に合わせる
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        # 縦長 - 高さに合わせる
        new_h = target_h
        new_w = int(target_h * img_ratio)
    
    # リサイズ
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 黒背景に配置
    result = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    result.paste(resized, offset)
    
    return result


def create_page_video(image: Image.Image, audio_path: str, 
                      duration: float) -> str:
    """画像と音声を結合してページ動画を作成"""
    # 画像をリサイズ
    resized_img = resize_image_for_video(image, OUTPUT_RESOLUTION)
    
    # 一時ファイルに保存
    img_path = tempfile.mktemp(suffix='.png')
    resized_img.save(img_path)
    
    # 動画クリップ作成
    img_clip = ImageClip(img_path, duration=duration)
    audio_clip = AudioFileClip(audio_path)
    
    video = img_clip.set_audio(audio_clip)
    
    # 一時ファイルに保存
    output_path = tempfile.mktemp(suffix='.mp4')
    video.write_videofile(
        output_path, 
        fps=OUTPUT_FPS, 
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    
    # クリーンアップ
    img_clip.close()
    audio_clip.close()
    os.remove(img_path)
    
    return output_path


def merge_videos(video_paths: list, output_path: str):
    """複数の動画を結合"""
    clips = [VideoFileClip(path) for path in video_paths]
    final = concatenate_videoclips(clips, method="compose")
    
    final.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    
    # クリーンアップ
    for clip in clips:
        clip.close()
    final.close()


def upload_to_hf_dataset(video_path: str, hf_token: str, 
                         repo_id: str) -> str:
    """Hugging Faceデータセットにアップロード"""
    api = HfApi()
    
    # ファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pdf_movie_{timestamp}.mp4"
    
    # アップロード
    url = api.upload_file(
        path_or_fileobj=video_path,
        path_in_repo=f"videos/{filename}",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    
    return url


def process_pdf_to_movie(pdf_file, program_style_name: str, gemini_api_key: str,
                         hf_token: str, hf_repo_id: str,
                         progress=gr.Progress()) -> tuple:
    """
    メイン処理: PDFを動画に変換
    """
    if pdf_file is None:
        return None, "PDFファイルをアップロードしてください", ""

    # 環境変数またはユーザー入力からAPIキーを取得
    api_key = gemini_api_key or ENV_GEMINI_API_KEY
    token = hf_token or ENV_HF_TOKEN
    repo_id = hf_repo_id or ENV_HF_REPO_ID

    if not api_key:
        return None, "Gemini APIキーを入力してください（または環境変数 GEMINI_API_KEY を設定）", ""

    if not token or not repo_id:
        return None, "Hugging FaceのトークンとリポジトリIDを入力してください（または環境変数 HF_TOKEN, HF_REPO_ID を設定）", ""

    try:
        pdf_path = pdf_file.name
        program_style = PROGRAM_STYLES.get(program_style_name, PROGRAM_STYLES["1人ラジオ風"])
        
        progress(0.05, desc="PDFを分割中...")
        
        # PDFを分割
        chunks = split_pdf(pdf_path, PAGES_PER_CHUNK)
        total_pages = sum(len(pages) for _, pages in chunks)
        
        progress(0.1, desc=f"PDFを{len(chunks)}チャンクに分割完了（計{total_pages}ページ）")
        
        # 全ページの画像を取得
        all_images = pdf_to_images(pdf_path)
        
        # ナレーション台本を生成
        all_scripts = {}
        for i, (chunk_path, page_numbers) in enumerate(chunks):
            progress(0.1 + (0.3 * i / len(chunks)), 
                    desc=f"ナレーション台本生成中... チャンク {i+1}/{len(chunks)}")
            
            scripts = generate_narration_script(
                chunk_path, page_numbers,
                program_style,
                api_key
            )
            all_scripts.update(scripts)
            
            # 一時ファイル削除
            os.remove(chunk_path)
        
        progress(0.4, desc="音声生成中...")
        
        # 各ページの音声を生成
        page_data = []  # [(image, audio_path, duration), ...]
        
        for i, page_num in enumerate(range(1, total_pages + 1)):
            progress(0.4 + (0.4 * i / total_pages), 
                    desc=f"音声生成中... ページ {page_num}/{total_pages}")
            
            script = all_scripts.get(page_num)
            
            # TTS生成（1人か2人かで分岐）
            if program_style["speakers"] == 1:
                # 1人用
                narration = script if isinstance(script, str) else f"ページ{page_num}です。"
                host_config = program_style["speaker_config"]["host"]
                
                pcm_data = text_to_speech_single(
                    narration,
                    host_config["voice"],
                    program_style.get("tts_style", "自然に読み上げてください。"),
                    api_key
                )
            else:
                # 2人用マルチスピーカー
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
            
            # WAVに保存
            wav_path = tempfile.mktemp(suffix='.wav')
            save_pcm_to_wav(pcm_data, wav_path)
            
            # 音声処理（速度変換、無音追加）
            processed_path, duration = process_audio(
                wav_path, AUDIO_SPEED, SILENCE_BEFORE, SILENCE_AFTER
            )
            
            # 元のWAV削除
            os.remove(wav_path)
            
            page_data.append((all_images[page_num - 1], processed_path, duration))
        
        progress(0.8, desc="動画作成中...")
        
        # 各ページの動画を作成
        video_paths = []
        for i, (image, audio_path, duration) in enumerate(page_data):
            progress(0.8 + (0.15 * i / len(page_data)), 
                    desc=f"動画作成中... ページ {i+1}/{len(page_data)}")
            
            video_path = create_page_video(image, audio_path, duration)
            video_paths.append(video_path)
            
            # 音声ファイル削除
            os.remove(audio_path)
        
        progress(0.95, desc="動画を結合中...")
        
        # 動画を結合
        final_video_path = tempfile.mktemp(suffix='.mp4')
        merge_videos(video_paths, final_video_path)
        
        # 一時動画ファイル削除
        for path in video_paths:
            os.remove(path)
        
        progress(0.98, desc="Hugging Faceにアップロード中...")
        
        # HFにアップロード
        hf_url = upload_to_hf_dataset(final_video_path, token, repo_id)
        
        progress(1.0, desc="完了！")
        
        # ダウンロードリンクを生成
        download_link = final_video_path
        
        # スピーカー情報
        speakers_info = ""
        for role, info in program_style["speaker_config"].items():
            speakers_info += f"  - {info['name']} (Voice: {info['voice']})\n"
        
        status_msg = f"""
✅ 動画生成完了！

📊 処理情報:
- 総ページ数: {total_pages}
- 番組スタイル: {program_style_name}
- 話者数: {program_style["speakers"]}人
{speakers_info}- 速度: {AUDIO_SPEED}x

📁 保存先:
- HF Dataset: {hf_url}
"""
        
        return final_video_path, status_msg, hf_url
        
    except Exception as e:
        import traceback
        error_msg = f"❌ エラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, ""


# ===========================
# Gradio インターフェース
# ===========================

with gr.Blocks(
    title="PDFtoMOVIEwithAUDIO",
    theme=gr.themes.Soft(),
    css="""
    .main-title {
        text-align: center;
        margin-bottom: 1em;
    }
    .status-box {
        min-height: 150px;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🎬 PDFtoMOVIEwithAUDIO
        
        PDFをナレーション付き動画に自動変換します。
        
        **処理フロー:**
        1. PDFを5ページごとに分割
        2. Gemini 3.0 Flash で番組スタイルに合わせた台本を自動生成
        3. Gemini TTS 2.5 Pro で音声生成（1人/2人対応）
        4. 音声を1.2倍速に変換し、前後に無音を追加
        5. 画像と音声を結合して動画化
        6. Hugging Face Datasetに自動保存
        """,
        elem_classes=["main-title"]
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 入力")
            
            pdf_input = gr.File(
                label="PDFファイル",
                file_types=[".pdf"],
                type="filepath"
            )
            
            program_style = gr.Dropdown(
                choices=list(PROGRAM_STYLES.keys()),
                value="1人ラジオ風",
                label="🎙️ 番組スタイル"
            )
            
            gr.Markdown("""
            **番組スタイル説明:**
            - 🎙️ **1人ラジオ風**: 親しみやすいDJが語りかける
            - 🎧 **2人ポッドキャスト風**: ホストとアシスタントの掛け合い
            - 😂 **2人漫才風**: ボケとツッコミで楽しく解説
            - 📺 **1人ニュース風**: 客観的で明確な報道スタイル
            - 🎓 **1人講義風**: 大学教授による丁寧な解説
            - 🎤 **2人インタビュー風**: 専門家への質問形式
            """)
            
            gr.Markdown("### 🔑 API設定")

            # 環境変数が設定されている場合は表示
            if ENV_GEMINI_API_KEY:
                gr.Markdown("✅ Gemini API Key: 環境変数から設定済み")
            gemini_key = gr.Textbox(
                label="Gemini API Key" + ("（オプション - 環境変数設定済み）" if ENV_GEMINI_API_KEY else ""),
                type="password",
                placeholder="環境変数 GEMINI_API_KEY から取得済み" if ENV_GEMINI_API_KEY else "AIza..."
            )

            if ENV_HF_TOKEN:
                gr.Markdown("✅ HF Token: 環境変数から設定済み")
            hf_token = gr.Textbox(
                label="Hugging Face Token" + ("（オプション - 環境変数設定済み）" if ENV_HF_TOKEN else ""),
                type="password",
                placeholder="環境変数 HF_TOKEN から取得済み" if ENV_HF_TOKEN else "hf_..."
            )

            hf_repo = gr.Textbox(
                label="HF Dataset Repository ID",
                value=ENV_HF_REPO_ID,
                placeholder="username/dataset-name"
            )
            
            generate_btn = gr.Button(
                "🎬 動画生成開始",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 📺 出力")
            
            video_output = gr.Video(
                label="生成された動画",
                interactive=False
            )
            
            status_output = gr.Textbox(
                label="ステータス",
                lines=12,
                interactive=False,
                elem_classes=["status-box"]
            )
            
            hf_url_output = gr.Textbox(
                label="🔗 HF Dataset URL",
                interactive=False
            )
    
    gr.Markdown(
        """
        ---
        ### 📝 使い方
        1. PDFファイルをアップロード
        2. お好みの番組スタイルを選択
        3. APIキーを入力
        4. 「動画生成開始」をクリック
        
        ### ⚠️ 注意事項
        - 処理時間はPDFのページ数に応じて変わります（1ページあたり約30秒〜1分）
        - Gemini APIの利用料金が発生します
        - 生成された動画はHF Datasetに自動保存されます
        - 2人スタイルはマルチスピーカーTTSを使用します
        """
    )
    
    generate_btn.click(
        fn=process_pdf_to_movie,
        inputs=[pdf_input, program_style, gemini_key, hf_token, hf_repo],
        outputs=[video_output, status_output, hf_url_output]
    )


if __name__ == "__main__":
    demo.launch()
