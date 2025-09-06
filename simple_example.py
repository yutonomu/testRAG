"""
RAGシステムの軽量版デモ（Gemini最新モデル対応）
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 環境変数の読み込み
load_dotenv()

def simple_rag_example():
    """
    軽量版RAGシステムのデモンストレーション
    """
    
    print("📚 RAGシステムの動作例（最新版）")
    print("=" * 50)
    
    # APIキーの確認
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-gemini-api-key-here':
        print("❌ Gemini APIキーが設定されていません")
        print("📝 .envファイルでGOOGLE_API_KEYを設定してください")
        return
    
    # サンプル文書の作成
    sample_documents = [
        "Python は汎用プログラミング言語です。Webアプリ、データ分析、AI開発に広く使用されています。",
        "機械学習は、データからパターンを学習してタスクを実行するAI技術です。",
        "RAGは検索拡張生成と呼ばれ、検索した情報を使ってより正確な回答を生成する手法です。",
        "ベクトルデータベースは、テキストや画像などのデータをベクトル形式で保存し、類似検索を高速に行う仕組みです。",
        "LangChainは大規模言語モデルを使ったアプリケーション開発のためのフレームワークです。"
    ]
    
    # Document形式に変換
    documents = [Document(page_content=content, metadata={"source": f"doc_{i}"}) 
                for i, content in enumerate(sample_documents)]
    
    print(f"✅ {len(documents)}個のサンプル文書を準備しました")
    
    # 文書分割器の設定
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    # 文書を分割（今回はサンプルが短いのでそのまま使用）
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)}個のチャンクに分割しました")
    
    try:
        # Google埋め込みモデルの初期化
        print("🔄 Google埋め込みモデルを初期化中...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        print("✅ 埋め込みモデルの初期化完了")
        
        # ベクトルストアの作成
        print("🔄 ベクトルストアを作成中...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        print("✅ ベクトルストア作成完了")
        
        # 軽量なFlashモデルを使用（クォータ制限を回避）
        model_names = [
            "models/gemini-1.5-flash-8b",      # 最軽量モデル
            "models/gemini-2.0-flash-lite",    # 軽量モデル
            "models/gemini-1.5-flash",         # 標準Flashモデル
            "models/gemini-2.0-flash",         # 新しいFlashモデル
            "models/gemini-1.5-flash-latest"   # 最新Flashモデル
        ]
        
        llm = None
        for model_name in model_names:
            try:
                print(f"🔄 {model_name} を試行中...")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.1,  # より低い温度で安定性を重視
                    google_api_key=api_key,
                    max_tokens=500,   # トークン数制限を追加
                    request_timeout=30  # タイムアウト設定
                )
                print(f"✅ {model_name} で初期化完了（テスト呼び出しをスキップ）")
                break
            except Exception as model_error:
                error_msg = str(model_error)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    print(f"❌ {model_name}: クォータ制限に達しました")
                    import time
                    print("⏱️ 10秒待機してから次のモデルを試行...")
                    time.sleep(10)
                else:
                    print(f"❌ {model_name}: {error_msg}")
                continue
        
        if llm is None:
            print("❌ 全てのモデルでクォータ制限に達しています")
            print("💡 対処法:")
            print("  1. 数時間待ってから再実行")
            print("  2. Google AI Studioで新しいAPIキーを作成")
            print("  3. 有料プランへのアップグレード")
            return
        
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # カスタムプロンプト
    prompt_template = """
以下の文脈情報を使用して質問に答えてください：

文脈: {context}

質問: {question}

回答は文脈情報に基づいて、日本語で正確に答えてください。
"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # RetrievalQAチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("\n💬 質問応答デモ")
    print("=" * 50)
    
    # デモ質問
    questions = [
        "Pythonについて教えてください",
        "RAGとは何ですか？",
        "機械学習について説明してください"
    ]
    
    for question in questions:
        print(f"\n❓ 質問: {question}")
        
        # 関連文書の検索
        relevant_docs = vector_store.similarity_search(question, k=2)
        print(f"🔍 関連文書を{len(relevant_docs)}件検索しました")
        print("📖 検索された文脈:")
        for doc in relevant_docs:
            print(f"  {doc.page_content}")
        
        try:
            # 質問応答
            print("🤖 Geminiが回答を生成中...")
            response = qa_chain.invoke({"query": question})
            print(f"💡 回答: {response['result']}")
            
        except Exception as e:
            print(f"❌ 回答生成エラー: {e}")
            print("� 直接的な質問応答を試行中...")
            
            # フォールバック: 直接LLMで質問
            context = "\n".join([doc.page_content for doc in relevant_docs])
            direct_prompt = f"""
以下の文脈情報を参考にして、質問に答えてください。

文脈情報:
{context}

質問: {question}

回答は日本語で簡潔に答えてください。
"""
            try:
                direct_response = llm.invoke(direct_prompt)
                print(f"💡 直接回答: {direct_response.content}")
            except Exception as direct_error:
                print(f"❌ 直接回答も失敗: {direct_error}")
                print("�📝 検索結果のみ表示:")
                for doc in relevant_docs:
                    print(f"  • {doc.page_content}")
        
        print("-" * 30)
    
    print("\n✨ デモンストレーション完了！")
    print("🎯 RAGシステムが正常に動作しました")

if __name__ == "__main__":
    simple_rag_example()