The multimodal (PDF) RAG parts below were taken from https://medium.com/@avneesh.khanna/agentic-rag-solution-for-llms-which-can-understand-pdfs-with-mutliple-images-and-diagrams-b154eea5f022#de30

---


## llm-agentic-multimodal-rag
LLM Agentic solution to multimodal RAG for large, complex PDF files

## Installation

### Create and activate conda environment
`conda create --name llm-tutorial`

`conda activate llm-tutorial`

### Install packages
`pip install -r requirements.txt`

## Set-up LlamaCloud
1. Create an account with LlamaCloud and obtain your API key from https://www.llamaindex.ai/
2. Replace your API key in `.env` file

## Set-up Ollama
1. Download and install Ollama application from https://ollama.com/
2. Download and run Llama 3.2 vision model locally using the following command

```
ollama run llama3.2-vision:11b-instruct-q4_K_M
```

Above command should download the model and run it on your machine. You can test it by typing a random question in the command-line terminal after the command-run completes.

## Setup Jupyter Notebooks
You'll need a local setup to run Jupyter Notebooks on your machine. Use this link to set it up - https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/install.html. I have been using it with VS Code as my code editor.

## Run the code
Open the notebook `llm_pdf.ipynb` and run notebook cells sequentially from the start. Modify notebook cell content to experiment and change values as required.

For details on this tutorial, refer to this free Medium article - [Agentic RAG solution for LLMs which can understand PDFs with multiple images and diagrams](https://medium.com/@avneesh.khanna/agentic-rag-solution-for-llms-which-can-understand-pdfs-with-mutliple-images-and-diagrams-b154eea5f022)
