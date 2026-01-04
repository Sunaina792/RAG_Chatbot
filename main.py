# 1. Imports & Env
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 2. Load Documents
def load_documents():
    loader = TextLoader("data/RAG_Notes.txt", encoding="utf-8")
    documents = loader.load()
    return documents

# 3. Chunk Documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

# 4. Create Vector Store (FAISS)
def create_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore

# 5. Retriever
def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    return retriever

# 6. Prompt Template (ANTI-HALLUCINATION)
def get_prompt():
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

# 7. Gemini LLM
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    return llm

# 8. RAG PIPELINE FUNCTION 
def answer_question(question, retriever, prompt, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )
    return response.content

# 9. Main Execution
if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    retriever = get_retriever(vectorstore)
    prompt = get_prompt()
    llm = load_llm()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        
        answer = answer_question(question, retriever, prompt, llm)
        print("\nAnswer:", answer)
