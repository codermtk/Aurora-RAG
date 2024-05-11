import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

memory = []


def preprocess_docs(docs):
    for doc in docs:
        doc.page_content = doc.page_content.strip()
    return docs


def load_and_retrieve_docs_from_url(url):
    loader = WebBaseLoader(web_paths=(url,), bs_kwargs=dict())
    docs = preprocess_docs(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)


    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()


def load_and_retrieve_docs_from_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    docs = preprocess_docs(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)


    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_memory(memory):
    return "\n\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in memory])


def rag_chain(url, pdf_file, question):
    global memory
    if url:
        retriever = load_and_retrieve_docs_from_url(url)
    elif pdf_file:
        retriever = load_and_retrieve_docs_from_pdf(pdf_file)
    else:
        return "Please provide a URL or a PDF file."
    
    retrieved_docs = retriever.invoke(question)  
    

    retrieved_docs = retrieved_docs[:5]

    formatted_context = format_docs(retrieved_docs)
    memory_context = format_memory(memory)
    full_context = f"{memory_context}\n\n{formatted_context}" if memory_context else formatted_context
    formatted_prompt = f"Based on the following context, answer the question as precisely as possible.\n\nContext: {full_context}\n\nQuestion: {question}"
    
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    answer = response['message']['content']
    

    memory.append({'question': question, 'answer': answer})
    
    return answer


iface = gr.Interface(
    fn=rag_chain,
    inputs=[gr.Textbox(label="URL"), gr.File(label="Elegir Archivo PDF"), gr.Textbox(label="Pregunta")],
    outputs=gr.Textbox(label="Respuesta"),
    title="Aurora RAG",
    description="Ingresa una URL o carga un PDF y haz una consulta para obtener respuestas de la cadena RAG"
)


iface.launch()

