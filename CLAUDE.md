# CLAUDE.md - Claude向けプロジェクトコンテキスト

このファイルはClaudeがプロジェクトを理解し、一貫した開発を行うためのコンテキストです。

---

## プロジェクト概要

**PDFtoMOVIEwithAUDIO** は、PDFをAIナレーション付き動画に変換するHugging Face Spaces Gradioアプリです。

### 主要技術
- **UI**: Gradio 5.9.1
- **AI**: Google Gemini API（スクリプト生成・TTS）
- **PDF処理**: PyMuPDF, pdf2image
- **動画生成**: moviepy, ffmpeg
- **ストレージ**: Hugging Face Hub

### Hugging Face リソース
- **HF Space**: `leave-everything/PDFtoMOVIEwithAUDIO`
- **HF Dataset**: `leave-everything/PDFtoMOVIEwithAUDIO`

---

## Claudeができること

### Git操作
- ファイルの作成・編集・削除
- コミットの作成
- `claude/` プレフィックスのブランチへのプッシュ
- ブランチの作成・切り替え
- git log, status, diff の確認

### できないこと（ユーザー操作が必要）
- `main` ブランチへの直接プッシュ（権限なし）
- PRのマージ（GitHub UIで実行）
- GitHub Secretsの設定

### プルリクエスト作成
1. 変更を `claude/` プレフィックスのブランチにプッシュ
2. PR作成URLを提供
3. ユーザーがGitHub UIでPRを作成・マージ

### 推奨ワークフロー
```
1. Claude: 変更をコミット
2. Claude: claude/xxx ブランチにプッシュ
3. Claude: PR作成URLを提供
4. ユーザー: GitHub UIでPRを作成・マージ
5. GitHub Actions: 自動でHF Spaceに同期
```

---

## GitHub ↔ HF Space 連携

### 設定済み
- `.github/workflows/sync-to-hf.yml` - 自動同期ワークフロー
- `HF_TOKEN` シークレット設定済み

### 動作
- mainブランチへのプッシュ時、自動でHF Spaceに同期
- 手動実行も可能（GitHub Actions → Run workflow）

---

## ファイル構成と役割

| ファイル | 役割 |
|---------|------|
| `app.py` | メインアプリケーション（全機能） |
| `requirements.txt` | Python依存関係 |
| `packages.txt` | システム依存関係（poppler, ffmpeg） |
| `README.md` | HF Spaces設定（YAML frontmatter） |
| `.github/workflows/sync-to-hf.yml` | GitHub Actions |
| `PLAN.md` | プロジェクト計画・ロードマップ |
| `STATUS.md` | 現在の開発状態 |
| `LOG.md` | 開発履歴（時系列） |
| `CLAUDE.md` | このファイル |

---

## app.py の構造

```
行 1-28:     インポート・初期化
行 29-141:   設定（PROGRAM_STYLES含む）
行 144-549:  コア処理関数
  - split_pdf()           : PDF分割
  - pdf_to_images()       : 画像抽出
  - generate_narration_script() : スクリプト生成
  - text_to_speech_single/multi() : TTS
  - process_audio()       : 音声処理
  - create_page_video()   : 動画生成
  - merge_videos()        : 動画結合
  - upload_to_hf_dataset(): HFアップロード
行 551-710:  メイン処理パイプライン
行 713-844:  Gradio UI定義
```

---

## 開発ガイドライン

### コーディング規約
1. **日本語コメント**: コメントは日本語で記述
2. **関数分離**: 各処理段階を独立した関数として実装
3. **エラーハンドリング**: try-exceptで適切にエラーを捕捉
4. **プログレス通知**: `gr.Progress`を使用して進捗を表示

### 変更時の注意
1. **PROGRAM_STYLES変更時**: 話者数（`speakers`）と`voice_config`の整合性を確認
2. **音声パラメータ変更時**: `AUDIO_SPEED`, `SILENCE_BEFORE/AFTER`の影響範囲を確認
3. **依存関係追加時**: `requirements.txt`と`packages.txt`の両方を確認
4. **gradioバージョン変更時**: `README.md`の`sdk_version`も更新

### ドキュメント更新ルール
- 機能追加・変更時 → `LOG.md`にエントリ追加
- 状態変更時 → `STATUS.md`を更新
- 計画変更時 → `PLAN.md`を更新

---

## 主要設定値

```python
# PDF処理
PAGES_PER_CHUNK = 5       # チャンクサイズ

# 音声
AUDIO_SPEED = 1.2         # 再生速度
SILENCE_BEFORE = 1000     # 前無音（ms）
SILENCE_AFTER = 500       # 後無音（ms）

# 動画
OUTPUT_FPS = 24           # フレームレート
OUTPUT_RESOLUTION = (1920, 1080)  # 解像度
```

---

## プログラムスタイル

6つのスタイルがあり、それぞれ異なるプロンプトと音声設定を持つ：

| キー | スタイル | 話者数 |
|-----|---------|--------|
| `radio_1` | 1人ラジオ風 | 1 |
| `podcast_2` | 2人ポッドキャスト風 | 2 |
| `manzai_2` | 2人漫才風 | 2 |
| `news_1` | 1人ニュース風 | 1 |
| `lecture_1` | 1人講義風 | 1 |
| `interview_2` | 2人インタビュー風 | 2 |

---

## 処理フロー

```
PDF Upload
    ↓
split_pdf() → 5ページ単位に分割
    ↓
pdf_to_images() → 各ページをPNG化
    ↓
generate_narration_script() → Geminiでスクリプト生成
    ↓
text_to_speech_single/multi() → TTS音声生成
    ↓
process_audio() → 速度調整・無音追加
    ↓
create_page_video() → ページ毎に動画作成
    ↓
merge_videos() → 全動画を結合
    ↓
upload_to_hf_dataset() → HFにアップロード
    ↓
Output: MP4 + HF URL
```

---

## よくある作業

### 新スタイル追加
1. `PROGRAM_STYLES`に新エントリ追加
2. `speakers`を1か2に設定
3. `system_prompt`でスタイルの特徴を記述
4. `voice_config`で使用する音声を指定
5. UIの説明文を更新

### 音声設定変更
1. `AUDIO_SPEED`で速度調整（1.0=等速）
2. `SILENCE_BEFORE/AFTER`で無音時間調整
3. 音声名は: Kore, Puck, Charon, Fenrir, Aoede, Alnilam

### デバッグ
1. `print()`文でログ出力（HF Spacesのログで確認可能）
2. 各関数の戻り値を確認
3. 一時ファイルの存在確認

### ビルドエラー対応
1. HF Spacesのビルドログを確認
2. 依存関係の競合をチェック（特にwebsockets）
3. `requirements.txt`を修正
4. 必要に応じて`README.md`の`sdk_version`を更新

---

## 禁止事項

- APIキーをコードにハードコーディングしない
- 一時ファイルを削除せずに残さない
- エラーを握りつぶさない（必ずログ出力）
- UI文言を英語にしない（日本語を維持）

---

## 参照ドキュメント

- [Gradio Docs](https://www.gradio.app/docs)
- [Google Gemini API](https://ai.google.dev/docs)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
- [MoviePy](https://zulko.github.io/moviepy/)

---

*このファイルはClaudeがプロジェクトを理解するための参照資料です*
