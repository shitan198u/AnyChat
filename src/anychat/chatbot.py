import streamlit as st
import subprocess

from langchain_core.messages import AIMessage, HumanMessage

from langchain_local import LangchainLocal
from uploadFile import UploadFile
from helper.helper import Helper
from ingest import GetVectorstore


def configure_api_key(api_key_name):
    # Configure the API key
    api_key = st.secrets[api_key_name]

    # Create an instance of the Helper class
    helper = Helper()

    # Check if the API key is set
    if api_key == "":
        # Prompt the user to enter the API key
        api_key = st.text_input(f"{api_key_name.capitalize()} API Key", type="password")

        # If the user enters an API key, write it to the secrets.toml file.
        if st.button("Set API_KEY") and api_key != "":
            helper.set_api_key(api_key_name, api_key)
            st.rerun()

    else:
        # If the API key is already set, display a message
        if not st.session_state.api_state_update:
            st.write("API key is set✅")

        # Provide an option to update the API key
        with st.expander(
            "Update API Key?",
        ):
            api_key = st.text_input("Enter API key", type="password", key="second")

            # If the user enters a new API key, update it in the secrets.toml file.
            if st.button("Confirm") and api_key != "":
                helper.set_api_key(api_key_name, api_key)
                # st.toast("API_KEY Updated✅")
                st.session_state.api_state_update = True
                # st.rerun()

    # Display a toast message when the API key is updated
    if st.session_state.api_state_update:
        st.toast("API_KEY Updated✅")
        st.session_state.api_state_update = False


#############################################################
def on_embedding_model_change(selected_model):
    if selected_model != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_model
        st.session_state.embedding_model_change_state = True


def select_embedding_model():
    if st.session_state.embedding_model == "Ollama Embeddings":
        model_name = "Ollama Embeddings"
    elif st.session_state.embedding_model == "GooglePalm Embeddings":
        model_name = "GooglePalm Embeddings"
    else:
        raise ValueError("Unknown embedding model")
    return model_name


def process_prompt():
    # Display chat history
    langchain_local = LangchainLocal(st.session_state)
    for message in st.session_state.chat_dialog_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.chat_dialog_history.append(HumanMessage(content=prompt))

        with st.chat_message("Human"):
            st.write(prompt)

        with st.chat_message("AI"):
            response = st.write_stream(
                langchain_local.get_response(
                    user_input=prompt,
                    chat_history=st.session_state.chat_dialog_history,
                    vectorstore=st.session_state.vectorstore,
                )
            )

        st.session_state.chat_dialog_history.append(AIMessage(content=response))


def load_models():
    LLM_TYPES = ["Groq", "Ollama", "Google"]
    EMBEDDING_MODELS = ["Ollama Embeddings", "GooglePalm Embeddings"]

    # Checking the available ollama models
    try:
        ollama_list_output = (
            subprocess.check_output(["ollama", "list"]).decode().split("\n")
        )
    except Exception:
        try:
            ollama_list_output = (
                subprocess.check_output(
                    ["docker", "exec", "-it", "ollama", "ollama", "list"]
                )
                .decode()
                .split("\n")
            )
        except Exception:
            ollama_list_output = []

    OLLAMA_MODELS = [
        line.split()[0]
        for line in ollama_list_output
        if ":" in line and "ollama:" not in line
    ]

    # Define Groq models
    GROQ_MODELS = ["mixtral-8x7b-32768", "llama2-70b-4096"]

    model_type = st.selectbox("Select LLM ⬇️", LLM_TYPES)
    if model_type == "Google":
        # configure_google_palm()
        configure_api_key("palm_api_key")
    elif model_type == "Ollama":
        if not OLLAMA_MODELS:
            st.error(
                "Ollama is not configured properly, Make sure:\n\n"
                "1. You have installed Ollama.\n"
                "2. Ollama is running.\n"
                "3. You have downloaded an Ollama model like Mistral 7B."
            )
            st.session_state.error = True
        else:
            st.session_state.ollama_model = st.selectbox("Ollama Model", OLLAMA_MODELS)
    elif model_type == "Groq":
        st.session_state.groq_model = st.selectbox("Groq Model", GROQ_MODELS)
        # configure_groq_api()
        configure_api_key("groq_api_key")
    st.session_state.llm_type = model_type
    # handling the embedding models
    embedding_model = st.radio("Embedding Model ⬇️", EMBEDDING_MODELS)
    on_embedding_model_change(embedding_model)


def initialize_ui():
    st.set_page_config(
        page_title="Any Chat",
        page_icon=":bee:",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("Any Chat :bee: :leaves: ")
    initialize_session_state()


def initialize_session_state():
    # checking the session state for the conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_dialog_history" not in st.session_state.keys():
        st.session_state.chat_dialog_history = [
            AIMessage(content="Hello! How can I help you today?"),
        ]
    # using for the check of uploaded documents
    if "disabled" not in st.session_state:
        st.session_state.disabled = True
    # using for api state check
    if "api_state_update" not in st.session_state:
        st.session_state.api_state_update = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = " "
    if "llm_type" not in st.session_state:
        st.session_state.llm_type = "Google"
    if "embedding_model_change_state" not in st.session_state:
        st.session_state.embedding_model_change_state = False
    if "error" not in st.session_state:
        st.session_state.error = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = ""


def process_documents():
    if st.session_state.disabled:
        st.write("🔒 Please upload and process your Documents to unlock the question field.")
        load_models()
        upload_and_process_files()
    else:
        process_prompt()


def upload_and_process_files():
    documents = st.file_uploader(
        "Upload the PDFs here:",
        accept_multiple_files=True,
        type=["xlsx", "xls", "csv", "pptx", "docx", "pdf", "txt"],
    )
    if documents and st.button(
        "Process",
        type="secondary",
        use_container_width=True,
        disabled=st.session_state.error,
    ):
        st.toast(
            """Hang tight! the documents are being processed for you,
                it might take several minutes depending on the size of your documents""",
            icon="🤖",
        )
        with st.spinner("Processing..."):
            if (
                st.session_state.disabled
                or st.session_state.embedding_model_change_state
            ):
                process_uploaded_documents(documents)
                st.session_state.disabled = False
                st.rerun()


def process_uploaded_documents(documents):
    text_chunks = []
    for docs in documents:
        upload = UploadFile(docs)
        splits = upload.get_document_splits()
        text_chunks.extend(splits)
    model_name = select_embedding_model()
    get_vectorstore_instance = GetVectorstore()
    st.session_state.vectorstore = get_vectorstore_instance.get_vectorstore(
        text_chunks, model_name
    )
    # st.session_state.vectorstore = get_vectorstore(text_chunks)
    st.session_state.embedding_model_change_state = False
    # retriever_chain = get_context_retriever_chain(st.session_state.vectorstore)
    # st.session_state.conversation = get_conversational_rag_chain(retriever_chain)
    # Delete all files inside the temp folder
    helper = Helper()
    helper.deleteFilesInTemp()


def main():
    initialize_ui()
    process_documents()


if __name__ == "__main__":
    main()
