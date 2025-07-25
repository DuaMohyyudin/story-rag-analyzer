import os
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama  # Will replace with fallback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class StoryRAG:
    def __init__(self, pdf_path="wish_dragon.pdf"):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.llm_ready = False
        self.load_story()  # Load story first
        
    def load_story(self):
        """Load and process the PDF story"""
        try:
            # 1. Check if PDF exists
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"Missing PDF: {self.pdf_path}")
            
            # 2. Load and split PDF
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)
            
            # 3. Create embeddings (updated HuggingFace import)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 4. Create vector store
            self.vectorstore = FAISS.from_documents(splits, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # 5. Try Ollama, fallback to smaller model if needed
            self._setup_llm_with_fallback()
            
            print(f"✓ Story loaded: {self.pdf_path}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise

    def _setup_llm_with_fallback(self):
        """Try Ollama first, fallback to smaller local model"""
        try:
            # Attempt Ollama
            self.llm = Ollama(model="mistral", temperature=0.3)
            self.llm.invoke("test")  # Test connection
            self.llm_ready = True
            print("✓ Using Ollama (mistral)")
        except Exception as e:
            print(f"⚠️ Ollama not available: {str(e)}")
            print("• Run: `ollama pull mistral` if you want to use Ollama")
            print("• Falling back to smaller local model...")
            self._setup_fallback_llm()

    def _setup_fallback_llm(self):
        """Fallback using CTransformers"""
        try:
            from langchain_community.llms import CTransformers
            self.llm = CTransformers(
                model="TheBloke/zephyr-7B-beta-GGUF",
                model_file="zephyr-7b-beta.Q2_K.gguf",
                model_type="mistral",
                config={'max_new_tokens': 128}
            )
            self.llm_ready = True
            print("✓ Using fallback model (zephyr-7b, 3MB)")
        except Exception as e:
            print(f"❌ Fallback failed: {str(e)}")
            self.llm_ready = False

    def ask_question(self, question):
        """Get answer about the story"""
        if not self.llm_ready:
            return "System not ready (LLM failed to initialize)"
        
        template = """Answer based on this story context:
        {context}
        
        Question: {question}
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return chain.invoke(question)
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    print("""
    Wish Dragon Story Analyzer
    -------------------------
    Initializing... (This may take 1-2 minutes)
    """)
    
    try:
        rag = StoryRAG("wish_dragon.pdf")
        
        if not rag.llm_ready:
            print("\n⚠️ Warning: Using limited fallback model. For better results:")
            print("1. Install Ollama: https://ollama.ai/download")
            print("2. Run: `ollama pull mistral`")
        
        print("""
        Ready! Ask questions like:
        - Who is the main character?
        - What is the dragon's power?
        - Type 'exit' to quit
        """)
        
        while True:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() == 'exit':
                break
                
            answer = rag.ask_question(user_input)
            print("\nAnswer:", answer)
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")

if __name__ == "__main__":
    # Install required packages if missing
    try:
        import langchain_huggingface
    except ImportError:
        print("Installing required packages...")
        os.system("pip install langchain-huggingface ctransformers pypdf")
    
    main()