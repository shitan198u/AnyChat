# AnyChat: Chat with Your Documents

AnyChat is a powerful chatbot that allows you to interact with your documents (PDF, TXT, DOCX, ODT, PPTX, CSV, etc.) in a natural and conversational way. It leverages the capabilities of LangChain, Ollama, Groq, Gemini, and Streamlit to provide an intuitive and informative experience.

[Video Demo](https://github.com/shitan198u/AnyChat/assets/74671269/6cdaf9ef-1b52-4ddc-bb45-721b3886f826)


## Features

- **Conversational Interaction:** Ask questions about your documents and receive human-like responses from the chatbot.
- **Multi-Document Support:** Upload and process various document formats, including PDFs, text files, Word documents, spreadsheets, and presentations.
- **Website-Chat Support:** Chat with any valid website.
- **Advanced Language Models:** Choose from different language models (LLMs) like Ollama, Groq, and Gemini to power the chatbot's responses.
- **Embedding Models:** Select from Ollama Embeddings or GooglePalm Embeddings to enhance the chatbot's understanding of your documents.
- **User-Friendly Interface:** Streamlit provides a clean and intuitive interface for interacting with the chatbot.

## Installation

### Prerequisites

- Python 3.10 or higher
- A virtual environment (recommended)

### Clone the Repository

Clone the AnyChat repository from GitHub:

```bash
git clone https://github.com/shitan198u/AnyChat.git
```
### Navigate to the working directory

```bash
cd Anychat
```

### Using `Rye` (Recommended)

1. Install the Rye package manager: [Installation Guide](https://rye-up.com/guide/installation/)

2. Sync the project:

```bash
rye sync
```

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

## Configuration

- Rename the `secrets_example.toml` file to `secrets.toml` in the `src/anychat/.streamlit/` directory.

## Ollama Installation

To use Ollama for AnyChat, you need to install Ollama and download the necessary models. Follow the instructions below:

1. **Install Ollama:**

Visit the official Ollama website for installation instructions: [Ollama Download](https://ollama.com/download)

2. **Download Ollama Models:**

Open your terminal and run the following commands to download the required models:

```bash
ollama pull nomic-embed-text
```

This command downloads the `nomic-embed-text` model, which is necessary for running Ollama embeddings.

```bash
ollama pull openchat
```

This command downloads the `openchat` model, which you can use as a language model in AnyChat.

## Usage

1. **Set API Keys:**
- If you're using Google Gemini or Groq, obtain the necessary API keys and store them securely in the `src/anychat/.streamlit/secrets.toml` file or Upload them in the chatbot interface.

2. **Run the Application:**

```bash
cd src/anychat
streamlit run chatbot.py
```
3. **Using Rye**

```bash
cd src/anychat
rye run streamlit run chatbot.py
```

4. **Upload Documents:**
- In the Streamlit interface, upload the documents you want to chat with.
- Click the "Process" button to process the documents.

5. **Start Chatting:**
- Once the documents are processed, you can start asking questions in the chat input field.
- The chatbot will analyze your documents and provide relevant answers based on their content.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Additional Notes

- For optimal performance, ensure you have the necessary resources (CPU, RAM) to handle the document processing and LLM computations.
- The chatbot's accuracy and responsiveness may vary depending on the complexity of your documents and the chosen LLM.
- Consider using a GPU-enabled environment if you have access to one, as it can significantly speed up the processing.
