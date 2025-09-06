# RAGシステム セットアップガイド

## 📋 概要
このプロジェクトは、LangChainとGemini APIを使用したRAG（Retrieval-Augmented Generation）システムです。
PDFファイルをアップロードして、その内容に基づいて質問応答ができます。

## 🔑 重要：APIキーの設定

### 1. Gemini APIキーの取得
1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. Googleアカウントでログイン
3. 「Create API Key」をクリック
4. APIキーをコピー

### 2. 環境変数の設定
`.env`ファイルを開き、以下の行を編集してください：

```bash
# 🔑 ここにあなたのAPIキーを貼り付けてください
GOOGLE_API_KEY=your-gemini-api-key-here
```

**実際のAPIキーに置き換えてください！**

## 🚀 セットアップ手順

### 1. 仮想環境の作成（推奨）
```bash
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# または
rag_env\Scripts\activate     # Windows
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. アプリケーションの起動
```bash
streamlit run rag_system.py
```

## 📖 使用方法

1. **APIキーの確認**: アプリ起動後、「APIキー設定の確認」で設定状況を確認
2. **ファイルアップロード**: 左サイドバーからPDFファイルをアップロード
3. **文書処理**: システムが自動的に文書を処理（数分かかる場合があります）
4. **質問入力**: 処理完了後、質問を入力して回答を取得

## 🔧 RAGシステムの構成要素

### 主要コンポーネント
- **文書ローダー**: PDFファイルの読み込み
- **テキスト分割器**: 文書を検索可能なチャンクに分割
- **埋め込みモデル**: テキストをベクトル化
- **ベクトルストア**: 検索用のデータベース
- **LLMモデル**: Gemini Pro（回答生成）

### 処理フロー
1. PDF → テキスト抽出
2. テキスト → チャンク分割
3. チャンク → ベクトル埋め込み
4. 質問 → 関連チャンク検索
5. 検索結果 + 質問 → Gemini → 回答

## ⚠️ トラブルシューティング

### APIキーエラー
- `.env`ファイルのAPIキーが正しく設定されているか確認
- APIキーに余分なスペースや改行がないか確認

### メモリエラー
- 大きなPDFファイルの場合、チャンクサイズを小さくする
- `rag_system.py`の`chunk_size`を500に変更

### インストールエラー
- Python 3.8以上を使用
- 仮想環境を使用することを推奨

## 📁 ファイル構成
```
testRAG/
├── requirements.txt    # 依存関係
├── .env               # 環境変数（APIキー）
├── rag_system.py      # メインアプリケーション
└── README.md          # このファイル
```
