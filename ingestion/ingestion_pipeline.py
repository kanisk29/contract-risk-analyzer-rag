import warnings
warnings.filterwarnings("ignore")

import os 
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma

def loader_files(doc_path = "data/law_files"):
    loader = DirectoryLoader(
        path=doc_path,
        glob = "*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}   
    )
    docs = loader.load()
    for i,doc in enumerate(docs):
        print(f"Document Name: {doc.metadata['source']}")
        print(f"Document Length: {len(doc.page_content)}")
    return docs

def chunker(docs): 
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
    doc_chunks = txt_splitter.split_documents(docs)
    for chunk in doc_chunks:
        source = chunk.metadata.get("source", "")
        filename = os.path.basename(source)
        chunk.metadata["source_file"] = filename
        chunk.metadata["clause_type"] = filename.replace(".txt", "")
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(doc_chunks)} chunks")
    return doc_chunks

def create_vector_stores(doc_chunks):
    for doc in doc_chunks:
        doc.page_content = doc.page_content.strip()
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    persist_directory = 'db/chroma_db'
    embedding_function = embeddings
    vecdb = Chroma.from_documents(collection_metadata={"hnsw:space" : "cosine"},documents=doc_chunks,persist_directory=persist_directory,embedding=embedding_function)
    print("Saved to: ",persist_directory)
    return vecdb

def run_ingestion():
    docs = loader_files()
    doc_chunks = chunker(docs)
    vecdb = create_vector_stores(doc_chunks)
    return vecdb

run_ingestion()