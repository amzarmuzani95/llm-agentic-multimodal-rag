import os
# import torch
# import requests
import nest_asyncio
from tqdm import tqdm
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker

from llm_pdf_functions import get_text_nodes, MultimodalQueryEngine


load_dotenv()

LLM_MODEL = "llama3.2-vision:11b" # # set LLama3.2-11b-visions as Ollama model
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
DOCS_PATH = "data/documents" # specify file path of docs
VECTOR_MODEL = "BAAI/bge-small-en-v1.5" # set BAAI/bge-small-en-v1.5 as vector store embedding model 

# replace PDF file of interest
# pdf_file = "RatingScales_YBOCS_m.pdf"

print("Creating LLM Model")
llm_model=Ollama(model=LLM_MODEL, request_timeout=500)

print("Creating Vector Model")
vector_store_embedding = HuggingFaceEmbedding(model_name=VECTOR_MODEL)

# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
nest_asyncio.apply()

# Parsing raw pdf using LlamaParse for getting Json Structured Output
print("Creating LlamaParse objects")
not_from_cache = False
parser_txt = LlamaParse(verbose=True, invalidate_cache=not_from_cache, result_type="text") # type: ignore
parser_md = LlamaParse(verbose=True, invalidate_cache=not_from_cache, result_type="markdown") # type: ignore

# for file in DOCS_PATH:
#     print(f"Parsing {file}")
#     docs_text = parser_txt.load_data(file)
#     print(f"Parsing PDF file...")
#     md_json_objs = parser_md.get_json_result(file)
#     md_json_list = md_json_objs[0]["pages"]

#     # extract images as a dict from parser
#     image_dicts = parser_md.get_images(md_json_objs, download_path="llm_images")

docs_text = []
md_json_objs =[]
image_dicts = []

for file in tqdm(os.listdir(DOCS_PATH)):
    if file.endswith(".pdf"):
        print(f"Parsing text from {file}...")
        docs_text += parser_txt.load_data(os.path.join(DOCS_PATH, file))
        print(f"Parsing PDF file {file}...")
        md_json_objs += parser_md.get_json_result(os.path.join(DOCS_PATH, file))
        image_dicts += parser_md.get_images(md_json_objs[-1], download_path="llm_images")

# this will split into pages
text_nodes = get_text_nodes(docs_text, json_dicts=md_json_objs, image_dicts=image_dicts)

## Build Index
# Once the text nodes are ready, we feed into our vector store index abstraction, which will index these nodes into a simple in-memory vector store
print("Creating index")
index = None
if not os.path.exists("storage_nodes"):
    index = VectorStoreIndex(text_nodes, embed_model=vector_store_embedding) # type: ignore
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist("./storage_nodes")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage_nodes")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index", embed_model=vector_store_embedding)

### Build Multimodal Query Engine
# We now use LlamaIndex abstractions to build a custom query engine. In contrast to a standard RAG query engine that will retrieve the text node and only put that into the prompt 
# (response synthesis module), this custom query engine will also load the image document, and put both the text and image document into the response synthesis module.

query_engine = MultimodalQueryEngine(
    retriever=index.as_retriever(similarity_top_k=5), multi_modal_llm=llm_model
)

print("Building a Multimodal Agent")
llm_model_tool_calling=Ollama(model="llama3.2:1b")

# Tool for querying the engine to retrieve contextual information around user query
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="query_engine_tool",
    description=(
        "Useful for retrieving specific context from the data. Do NOT select if question asks for a summary of the data."
    ),
)

# Set-up the agent for calling query engine tools
agent = FunctionCallingAgentWorker.from_tools(
        [query_engine_tool], llm=llm_model_tool_calling, verbose=True
        ).as_agent()

if __name__ == "__main__":
    query = (
        "What are compulsions?"
    )
    response = agent.query(query)
    print(response)
