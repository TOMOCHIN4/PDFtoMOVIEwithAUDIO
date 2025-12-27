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

PDFをナレーション付き動画に自動変換するアプリケーション。

## 機能
- PDFを5ページごとに分割して処理
- Gemini 2.0 Flashで番組スタイルに合わせた台本を自動生成
- Gemini TTS 2.5 Flashで音声生成（1人/2人対応）
- 音声を1.2倍速に変換し、前後に無音を追加
- 画像と音声を結合して動画化
- Hugging Face Datasetに自動保存

## 番組スタイル
- 1人ラジオ風
- 2人ポッドキャスト風
- 2人漫才風
- 1人ニュース風
- 1人講義風
- 2人インタビュー風

## 必要な環境変数
- `GEMINI_API_KEY`: Google Gemini APIキー
- `HF_TOKEN`: Hugging Faceトークン
- `HF_REPO_ID`: アップロード先リポジトリ（オプション）
