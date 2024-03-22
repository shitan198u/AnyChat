# AnyChat: Chat with Your Documents

AnyChat is a powerful chatbot that allows you to interact with your documents (PDF, TXT, DOCX, ODT, PPTX, CSV, etc.) in a natural and conversational way. It leverages the capabilities of LangChain, Ollama, Groq, Gemini, and Streamlit to provide an intuitive and informative experience.

[Video Demo](https://github.com/shitan198u/AnyChat/assets/74671269/bf254a76-8e47-4d8f-a4d7-03a318d252d6)

## Features

- **Conversational Interaction:** Ask questions about your documents and receive human-like responses from the chatbot.
- **Multi-Document Support:** Upload and process various document formats, including PDFs, text files, Word documents, spreadsheets, and presentations.
- **Advanced Language Models:** Choose from different language models (LLMs) like Ollama, Groq, and Gemini to power the chatbot's responses.
- **Embedding Models:** Select from Ollama Embeddings or GooglePalm Embeddings to enhance the chatbot's understanding of your documents.
- **User-Friendly Interface:** Streamlit provides a clean and intuitive interface for interacting with the chatbot.

## Installation

### Prerequisites

- Python 3.10 or higher
- A virtual environment (recommended)

### Using `venv`

1. Create a virtual environment:

```bash
python -m venv anychat-env
```

2. Activate the virtual environment:

```bash
source anychat-env/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Using `conda`

1. Create a conda environment:

```bash
conda create -n anychat-env python=3.12
```

2. Activate the conda environment:

```bash
conda activate anychat-env
```

3. Install the required dependencies:

```bash
conda install --file requirements.txt
```

## Usage

1. **Set API Keys:**
- If you're using Google Palm or Groq, obtain the necessary API keys and store them securely in the `src/anychat/.streamlit/secrets.toml` file.

2. **Run the Application:**

```bash
streamlit run src/anychat/chatbot.py
```

3. **Upload Documents:**
- In the Streamlit interface, upload the documents you want to chat with.
- Click the "Process" button to process the documents.

4. **Start Chatting:**
- Once the documents are processed, you can start asking questions in the chat input field.
- The chatbot will analyze your documents and provide relevant answers based on their content.

## Configuration

- You can choose different LLMs and embedding models in the Streamlit interface to customize the chatbot's behavior.
- Refer to the `src/anychat/chatbot.py` file for more configuration options.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

## Additional Notes

- For optimal performance, ensure you have the necessary resources (CPU, RAM) to handle the document processing and LLM computations.
- The chatbot's accuracy and responsiveness may vary depending on the complexity of your documents and the chosen LLM.
- Consider using a GPU-enabled environment if you have access to one, as it can significantly speed up the processing.
