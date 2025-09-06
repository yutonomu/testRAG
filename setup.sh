#!/bin/bash

# ================================
# RAGシステム セットアップスクリプト
# ================================

echo "🚀 RAGシステムのセットアップを開始します"

# Python仮想環境の作成
echo "📦 Python仮想環境を作成中..."
python3 -m venv rag_env

# 仮想環境の有効化
echo "✅ 仮想環境を有効化中..."
source rag_env/bin/activate

# 依存関係のインストール
echo "📚 必要なライブラリをインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎉 セットアップが完了しました！"
echo ""
echo "================================"
echo "🔑 重要: 次の手順を実行してください"
echo "================================"
echo "1. .envファイルを開く"
echo "2. GOOGLE_API_KEY=your-gemini-api-key-here の部分を"
echo "   実際のGemini APIキーに置き換える"
echo "3. APIキーの取得: https://makersuite.google.com/app/apikey"
echo ""
echo "================================"
echo "🚀 アプリケーションの起動方法"
echo "================================"
echo "仮想環境を有効化:"
echo "  source rag_env/bin/activate"
echo ""
echo "Streamlitアプリを起動:"
echo "  streamlit run rag_system.py"
echo ""
echo "シンプルな例を実行:"
echo "  python simple_example.py"
echo ""
