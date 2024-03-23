from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.manager import CallbackManager
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
import psutil
from streamlit import secrets


class LangchainLocal:
    def __init__(self, session_state):
        self.session_state = session_state
        self.llm = self.get_llm()

    def get_context_retriever_chain(self, vector_store):
        retriever = vector_store.as_retriever()

        # Create a prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                (
                    "human",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, keep it concise",
                ),
            ]
        )

        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

        return retriever_chain

    def get_conversational_rag_chain(self, retriever_chain):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        stuff_documents_chain = create_stuff_documents_chain(self.llm, prompt)

        return create_retrieval_chain(retriever_chain, stuff_documents_chain)

    def get_llm(self):
        llm_type = self.session_state.llm_type
        if llm_type == "Ollama":
            model = self.session_state.ollama_model
            llm = ChatOllama(
                base_url="http://localhost:11434",
                model=model,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=0.2,
                num_ctx=512,
                num_thread=max(1, int(psutil.cpu_count() * 0.9)),
                stream=True,
            )
        elif llm_type == "Google":
            google_api_key = secrets["palm_api_key"]
            llm = ChatGoogleGenerativeAI(
                # model="gemini-pro",
                model="models/gemini-1.0-pro-latest",
                google_api_key=google_api_key,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                # stream=True,
                max_output_tokens=4096,
                top_p=0.7,
                top_k=30,
                temperature=0,
                verbose=True,
                convert_system_message_to_human=True,
            )
        elif llm_type == "Groq":
            groq_api_key = secrets["groq_api_key"]
            model = self.session_state.groq_model
            llm = ChatGroq(
                model=model,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                api_key=groq_api_key,
                temperature=0,
                streaming=True,
            )
        else:
            raise ValueError("Unknown LLM model")

        return llm

    def get_response(self, user_input, chat_history, vectorstore):
        retriever_chain = self.get_context_retriever_chain(vectorstore)
        conversational_rag_chain = self.get_conversational_rag_chain(retriever_chain)

        response_stream = conversational_rag_chain.stream(
            {"chat_history": chat_history, "input": user_input}
        )

        for chunk in response_stream:
            content = chunk.get("answer", "")
            yield content
