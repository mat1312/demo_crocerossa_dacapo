"""
Script per indicizzare i documenti della Croce Rossa Italiana nel vector database.
Questo script processa i file dalla cartella 'output' e li indicizza nel vector database.
"""

import os
import glob
import argparse
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurazione percorsi
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Cartella dei documenti originali
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Configurazione chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_documents(input_dir: str) -> List[Document]:
    """Carica tutti i documenti dalla directory specificata."""
    documents = []
    
    # Verifica che la directory esista
    if not os.path.exists(input_dir):
        logger.error(f"La directory {input_dir} non esiste.")
        return []
    
    # Carica i PDF
    pdf_pattern = os.path.join(input_dir, "**/*.pdf")
    for pdf_path in glob.glob(pdf_pattern, recursive=True):
        logger.info(f"Caricamento PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    # Carica i file di testo
    txt_pattern = os.path.join(input_dir, "**/*.txt")
    for txt_path in glob.glob(txt_pattern, recursive=True):
        logger.info(f"Caricamento TXT: {txt_path}")
        loader = TextLoader(txt_path, encoding='utf-8')
        documents.extend(loader.load())
    
    # Carica i file markdown
    md_pattern = os.path.join(input_dir, "**/*.md")
    for md_path in glob.glob(md_pattern, recursive=True):
        logger.info(f"Caricamento MD: {md_path}")
        loader = UnstructuredMarkdownLoader(md_path)
        documents.extend(loader.load())
    
    # Carica altri formati potenzialmente utili (HTML, DOCX, ecc.)
    # È possibile aggiungere altri loader se necessario
    
    logger.info(f"Caricati {len(documents)} documenti in totale")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Divide i documenti in chunks più piccoli."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documenti suddivisi in {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: List[Document]):
    """Crea il vector store utilizzando Chroma."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    # Usa OpenAI per gli embeddings
    embeddings = OpenAIEmbeddings()
    
    # Crea il vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    # Persisti il vector store su disco
    vector_store.persist()
    logger.info(f"Vector store creato e salvato in {VECTOR_STORE_PATH}")
    return vector_store


def main():
    # Configurazione parser per argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Indicizzazione documenti per l\'assistente CRI')
    parser.add_argument('--input-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory contenente i documenti da indicizzare (default: {OUTPUT_DIR})')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    logger.info(f"Inizializzazione processo di indicizzazione documenti da {input_dir}")
    
    # Carica i documenti
    documents = load_documents(input_dir)
    if not documents:
        logger.warning(f"Nessun documento trovato nella directory {input_dir}")
        return
    
    # Dividi i documenti in chunks
    chunks = split_documents(documents)
    
    # Crea e salva il vector store
    create_vector_store(chunks)
    
    logger.info("Processo di indicizzazione completato con successo")


if __name__ == "__main__":
    main()