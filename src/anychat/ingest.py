from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit import secrets, error, stop
import psutil

from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader

# from langchain_community.document_loaders.parsers import OpenAIWhisperParser,OpenAIWhisperParserLocal
from langchain.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)


class FileProcessor:
    def __init__(self, fileLocation):
        self.fileLocation = fileLocation

    def process(self, contentType):
        # matching the file types for loaders
        if contentType == "text/plain":
            loader = TextLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "application/pdf":
            loader = PyPDFLoader(self.fileLocation)
            document = loader.load_and_split()
        elif (
            contentType
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            loader = Docx2txtLoader(self.fileLocation)
            document = loader.load()
        elif (
            contentType
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            loader = UnstructuredExcelLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "text/csv":
            loader = CSVLoader(self.fileLocation)
            document = loader.load()
        elif (
            contentType == "application/vnd.ms-powerpoint"
            or contentType
            == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ):
            loader = UnstructuredPowerPointLoader(self.fileLocation)
            document = loader.load()
        else:
            # for unsupported file type
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
        )
        chunks = text_splitter.split_documents(document)
        # chunks = text_splitter.create_documents(document)
        return chunks


class GetVectorstore:
    def get_vectorstore(self, chunks, model_name):
        if model_name == "Ollama Embeddings":
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text:latest",
                num_thread=max(1, int(psutil.cpu_count() * 0.9)),
            )
        elif model_name == "GooglePalm Embeddings":
            google_api_key = secrets["palm_api_key"]
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=google_api_key
            )
        else:
            raise ValueError("Unknown embedding model")

        if not chunks:
            return None
        
        vectorstore = Qdrant.from_documents(
            chunks,
            embeddings,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name="my_documents",
        )
        return vectorstore


class WebContentProcessor:
    def __init__(self, url):
        self.url = url

    def process(self):
        loader = WebBaseLoader(self.url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
        )
        chunks = text_splitter.split_documents(data)

        return chunks


class YouTubeChatIngest:
    def __init__(self, url, save_dir, local=False):
        self.url = url
        self.save_dir = save_dir
        self.local = local

    def load_data(self):
        if self.local:
            loader = GenericLoader(
                YoutubeAudioLoader([self.url], self.save_dir),
                OpenAIWhisperParserLocal(),
            )
        else:
            loader = GenericLoader(
                YoutubeAudioLoader([self.url], self.save_dir), OpenAIWhisperParser()
            )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
        )
        chunks = text_splitter.split_documents(docs)

        return chunks
