"""
RAGシステム実装 - LangChain + Gemini
このファイルはRAG（Retrieval-Augmented Generation）の基本実装です
"""

import os
import streamlit as st
from dotenv import load_dotenv

# LangChainライブラリのインポート
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==================================================
# 🔑 重要: 環境変数の読み込み（APIキー設定）
# ==================================================
# .envファイルからAPIキーを読み込みます
# GeminiのAPIキーが設定されていることを確認してください
load_dotenv()

class RAGSystem:
    """
    RAGシステムのメインクラス
    1. 文書の読み込み
    2. 文書の分割
    3. 埋め込みベクトル化
    4. ベクトルデータベースの構築
    5. 質問応答システム
    """
    
    def __init__(self):
        """
        RAGシステムの初期化
        - Gemini APIキーの確認
        - 埋め込みモデルの設定
        - LLMモデルの設定
        """
        
        # ==================================================
        # 🔑 重要: Gemini APIキーの確認と設定
        # ==================================================
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key == 'your-gemini-api-key-here':
            st.error("🚨 Gemini APIキーが設定されていません！")
            st.error(".envファイルでGOOGLE_API_KEYを設定してください")
            st.stop()
        
        # 埋め込みモデルの初期化（文書をベクトル化するため）
        # HuggingFaceの多言語対応モデルを使用
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Gemini LLMの初期化（質問応答のため）
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,  # 創造性と正確性のバランス
            google_api_key=api_key
        )
        
        # ベクトルストア（文書の検索データベース）
        self.vector_store = None
        
        # 文書分割器の設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # 1つのチャンクのサイズ（文字数）
            chunk_overlap=200,    # チャンク間のオーバーラップ
            length_function=len,  # 長さを測る関数
        )
    
    def load_and_process_documents(self, file_path):
        """
        PDFファイルを読み込んで処理する
        
        Args:
            file_path: PDFファイルのパス
            
        Returns:
            処理済みの文書チャンクリスト
        """
        
        # PDFローダーで文書を読み込み
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 文書を小さなチャンクに分割
        # これにより関連する情報を効率的に検索できる
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def create_vector_store(self, chunks):
        """
        文書チャンクからベクトルデータベースを作成
        
        Args:
            chunks: 分割された文書チャンクのリスト
        """
        
        # FAISSベクトルストアの作成
        # 各チャンクを埋め込みベクトルに変換して保存
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
    
    def setup_qa_chain(self):
        """
        質問応答チェーンのセットアップ
        検索した情報を使ってGeminiが回答を生成
        """
        
        if self.vector_store is None:
            raise ValueError("ベクトルストアが作成されていません")
        
        # カスタムプロンプトテンプレートの定義
        # 検索した情報を使って日本語で回答するように指示
        prompt_template = """
あなたは親切なアシスタントです。以下の文脈情報を使用して、質問に正確に答えてください。

文脈情報:
{context}

質問: {question}

回答は以下の点に注意してください：
- 文脈情報に基づいて回答する
- 情報が不十分な場合は、「提供された情報では十分に答えられません」と回答する
- 日本語で分かりやすく回答する

回答:
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQAチェーンの作成
        # 1. 質問に関連する文書を検索
        # 2. 検索した情報をもとにGeminiが回答を生成
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # すべての関連文書を一度に処理
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # 関連する上位3つの文書を取得
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def ask_question(self, question):
        """
        質問に対する回答を生成
        
        Args:
            question: ユーザーの質問
            
        Returns:
            RAGシステムからの回答
        """
        
        if self.qa_chain is None:
            raise ValueError("QAチェーンが設定されていません")
        
        # 質問を処理して回答を生成
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

def main():
    """
    Streamlitアプリのメイン関数
    ウェブUIでRAGシステムを操作
    """
    
    st.title("📚 RAGシステム - 文書質問応答")
    st.markdown("---")
    
    # ==================================================
    # 🔑 重要: APIキー設定の確認表示
    # ==================================================
    with st.expander("🔑 APIキー設定の確認", expanded=False):
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key and api_key != 'your-gemini-api-key-here':
            st.success("✅ Gemini APIキーが設定されています")
        else:
            st.error("❌ Gemini APIキーが設定されていません")
            st.markdown("""
            **設定手順:**
            1. [Google AI Studio](https://makersuite.google.com/app/apikey) でAPIキーを取得
            2. `.env`ファイルの`GOOGLE_API_KEY`を更新
            3. アプリを再起動
            """)
    
    # RAGシステムの初期化
    try:
        rag_system = RAGSystem()
    except Exception as e:
        st.error(f"システムの初期化に失敗しました: {e}")
        return
    
    # サイドバーでファイルアップロード
    st.sidebar.title("📄 文書アップロード")
    uploaded_file = st.sidebar.file_uploader(
        "PDFファイルをアップロードしてください",
        type=['pdf'],
        help="RAGシステムで質問応答する文書をアップロードします"
    )
    
    # ファイルが選択された場合の処理
    if uploaded_file is not None:
        
        # 一時ファイルに保存
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.sidebar.success("✅ ファイルがアップロードされました")
        
        # 文書処理のプログレス表示
        with st.spinner("📖 文書を処理しています..."):
            
            # 1. 文書の読み込みと分割
            st.info("1. 文書を読み込んで分割しています...")
            chunks = rag_system.load_and_process_documents("temp.pdf")
            st.success(f"✅ {len(chunks)}個のチャンクに分割しました")
            
            # 2. ベクトルデータベースの作成
            st.info("2. ベクトルデータベースを作成しています...")
            rag_system.create_vector_store(chunks)
            st.success("✅ ベクトルデータベースを作成しました")
            
            # 3. 質問応答システムのセットアップ
            st.info("3. 質問応答システムをセットアップしています...")
            rag_system.setup_qa_chain()
            st.success("✅ RAGシステムの準備が完了しました")
        
        st.success("🎉 RAGシステムが準備完了！質問を入力してください")
        
        # 質問入力フォーム
        st.markdown("---")
        st.subheader("💬 質問してください")
        
        question = st.text_input(
            "質問を入力してください:",
            placeholder="例: この文書の主な内容は何ですか？"
        )
        
        # 質問処理
        if st.button("🚀 回答を生成", type="primary"):
            if question.strip():
                with st.spinner("🤔 回答を生成しています..."):
                    try:
                        # RAGシステムで回答を生成
                        answer = rag_system.ask_question(question)
                        
                        # 回答の表示
                        st.markdown("### 📝 回答")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"回答の生成中にエラーが発生しました: {e}")
            else:
                st.warning("質問を入力してください")
        
        # 一時ファイルの削除
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
    
    else:
        # ファイルが選択されていない場合の説明
        st.info("📄 左側のサイドバーからPDFファイルをアップロードしてください")
        
        # RAGシステムの説明
        with st.expander("ℹ️ RAGシステムとは？", expanded=True):
            st.markdown("""
            **RAG (Retrieval-Augmented Generation)** は以下の流れで動作します：
            
            1. **文書の読み込み**: PDFファイルを読み込みます
            2. **文書の分割**: 文書を小さなチャンクに分割します
            3. **埋め込み化**: 各チャンクをベクトルに変換します
            4. **ベクトル検索**: 質問に関連するチャンクを検索します
            5. **回答生成**: 検索した情報をもとにGeminiが回答を生成します
            
            **使い方：**
            1. PDFファイルをアップロード
            2. システムが文書を処理するまで待機
            3. 質問を入力して回答を取得
            """)

if __name__ == "__main__":
    # ==================================================
    # 🔑 重要: 環境変数の読み込み
    # ==================================================
    # .envファイルからAPIキーを読み込みます
    load_dotenv()
    
    main()
