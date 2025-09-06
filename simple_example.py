"""
シンプルなRAGシステムの例
コマンドライン版での基本的な使用例
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# ==================================================
# 🔑 重要: 環境変数の読み込み
# ==================================================
load_dotenv()

def simple_rag_example():
    """
    簡単なRAGの動作例
    サンプルテキストを使用してRAGの基本的な流れを理解
    """
    
    # ==================================================
    # 🔑 重要: APIキーの確認
    # ==================================================
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-gemini-api-key-here':
        print("❌ Gemini APIキーが設定されていません！")
        print(".envファイルでGOOGLE_API_KEYを設定してください")
        return
    
    # サンプル文書の作成
    sample_texts = [
        "Python は汎用プログラミング言語です。Webアプリ、データ分析、AI開発に広く使用されています。",
        "LangChain は大規模言語モデル（LLM）を使ったアプリケーション開発のためのフレームワークです。",
        "RAG（Retrieval-Augmented Generation）は、外部知識を検索して生成AIの回答品質を向上させる手法です。",
        "ベクトルデータベースは、テキストや画像などのデータをベクトル形式で保存し、類似検索を高速に行う仕組みです。",
        "Gemini は Google が開発した大規模言語モデルで、テキスト生成、質問応答、コード生成などができます。"
    ]
    
    print("📚 RAGシステムの動作例")
    print("=" * 50)
    
    # 1. 文書をDocumentオブジェクトに変換
    documents = [Document(page_content=text) for text in sample_texts]
    print(f"✅ {len(documents)}個のサンプル文書を準備しました")
    
    # 2. 文書を分割（この例では既に短いので分割は簡単）
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)}個のチャンクに分割しました")
    
    # 3. 埋め込みモデルの初期化
    print("🔄 埋め込みモデルを初期化中...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("✅ 埋め込みモデルの初期化完了")
    
    # 4. ベクトルストアの作成
    print("🔄 ベクトルストアを作成中...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ ベクトルストア作成完了")
    
    # 5. LLMの初期化
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=api_key
    )
    print("✅ Gemini LLM初期化完了")
    
    # 6. 質問応答のデモ
    questions = [
        "Pythonについて教えてください",
        "RAGとは何ですか？",
        "LangChainの特徴は？"
    ]
    
    print("\n💬 質問応答デモ")
    print("=" * 50)
    
    for question in questions:
        print(f"\n❓ 質問: {question}")
        
        # 関連文書を検索
        relevant_docs = vector_store.similarity_search(question, k=2)
        print(f"🔍 関連文書を{len(relevant_docs)}件検索しました")
        
        # 検索した文書の内容を表示
        context = "\n".join([doc.page_content for doc in relevant_docs])
        print(f"📖 検索された文脈:\n{context}\n")
        
        # Geminiに質問（検索した情報を含める）
        prompt = f"""
以下の文脈情報を参考にして、質問に答えてください。

文脈情報:
{context}

質問: {question}

回答:
"""
        
        response = llm.invoke(prompt)
        print(f"🤖 回答: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    simple_rag_example()
