---
title: PDFtoMOVIEwithAUDIO
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# PDFtoMOVIEwithAUDIO

PDFをAIナレーション付き動画に自動変換するアプリケーション。

## 機能

- PDFを5ページごとに分割して処理
- **Gemini 3 Flash Preview** で番組スタイルに合わせた台本を自動生成
  - Pydantic構造化出力で安定したJSON生成
  - チャンク位置認識で一貫性のあるナレーション
- **Gemini 2.5 Flash TTS** で音声生成（1人/2人対応）
  - 早口・正確な日本語発音
  - レートリミット対応リトライ機構
- 音声を1.2倍速に変換し、前後に無音を追加
- HD画質（1280×720）で高速エンコード
- ffmpeg直接結合で動画を高速マージ
- Hugging Face Datasetに自動保存

## 番組スタイル

| スタイル | 話者数 | 説明 |
|---------|--------|------|
| 1人ラジオ風 | 1人 | 親しみやすいラジオパーソナリティ |
| 2人ポッドキャスト風 | 2人 | ホストとアシスタントの対話 |
| 2人漫才風 | 2人 | ボケとツッコミのコメディ |
| 1人ニュース風 | 1人 | フォーマルなニュースキャスター |
| 1人講義風 | 1人 | 教授による学術的解説 |
| 2人インタビュー風 | 2人 | 専門家へのQ&A形式 |

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| UI | Gradio 5.9.1 |
| 台本生成 | Gemini 3 Flash Preview（構造化出力） |
| 音声生成 | Gemini 2.5 Flash TTS |
| PDF処理 | PyMuPDF, pdf2image |
| 動画生成 | moviepy, ffmpeg |

## 設定

```python
PAGES_PER_CHUNK = 5          # PDF分割単位
AUDIO_SPEED = 1.2            # 再生速度
OUTPUT_RESOLUTION = (1280, 720)  # HD画質
OUTPUT_FPS = 24              # フレームレート
```

## 必要な環境変数

| 変数名 | 説明 | 必須 |
|--------|------|------|
| `GEMINI_API_KEY` | Google Gemini APIキー | ✅ |
| `HF_TOKEN` | Hugging Faceトークン | ✅ |
| `HF_REPO_ID` | アップロード先リポジトリ | オプション |

## 開発

GitHub: [TOMOCHIN4/PDFtoMOVIEwithAUDIO](https://github.com/TOMOCHIN4/PDFtoMOVIEwithAUDIO)

GitHub Actionsによる自動デプロイ設定済み。
