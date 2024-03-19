from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit import secrets
import psutil


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
        # index = self.__indexDocument(chunks)

        # return index


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

        vectorstore = Qdrant.from_documents(
            chunks,
            embeddings,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name="my_documents",
        )
        return vectorstore


# class GetVectorstore:
#     def __init__(self, model_name):
#         self.model_name = model_name
#         if model_name == "Ollama Embeddings":
#             self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", num_thread=max(1, int(psutil.cpu_count() * 0.9)))
#         elif model_name == "GooglePalm Embeddings":
#             google_api_key = st.secrets["palm_api_key"]
#             self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=google_api_key)
#         else:
#             raise ValueError("Unknown embedding model")

#     def get_vectorstore(self, chunks):
#         vectorstore = Qdrant.from_documents(
#             chunks,
#             self.embeddings,
#             location=":memory:",  # Local mode with in-memory storage only
#             collection_name="my_documents",
#         )
#         return vectorstore
