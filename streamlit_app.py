import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import tempfile


# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# App title
st.title("🤖 RAG Document Chatbot")
st.caption("Upload your document (PDF, DOCX, or TXT) and ask questions about its content")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# File uploader
uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT"
)

def load_and_process_document(uploaded_file):
    """Load and process the uploaded document"""
    with st.spinner("Processing document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Load document based on file type
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
            documents = loader.load()
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return retriever

def get_prompt():
    """Create the prompt template"""
    template = """
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def answer_question(question, retriever, prompt, llm):
    """Answer the question based on the document"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )
    return response.content

# Process uploaded file if available
if uploaded_file is not None and not st.session_state.document_loaded:
    try:
        st.session_state.retriever = load_and_process_document(uploaded_file)
        st.session_state.document_loaded = True
        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")

# Show chat interface if document is loaded
if st.session_state.document_loaded:
    # Initialize LLM
    @st.cache_resource
    def load_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    
    llm = load_llm()
    prompt = get_prompt()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about the document...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(user_input, st.session_state.retriever, prompt, llm)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Reset button
if st.session_state.document_loaded:
    if st.button("Upload a new document"):
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.document_loaded = False
        st.rerun()

# Show info if no document is loaded
if not st.session_state.document_loaded:
    st.info("👆 Please upload a document to start chatting")