from typing import Optional

from llama_index.llms.ollama import Ollama
from pathlib import Path
from llama_index.core.schema import TextNode
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode

from system_prompt import QA_PROMPT

def create_image_index(image_dicts):
    """
    Create a dictionary which maps page numbers to image paths with the following format:

    {
        1: [Path("path/to/image1"), Path("path/to/image2")],    
        2: [Path("path/to/image3"), Path("path/to/image4")],
    }
    """
    image_index = {}

    for image_dict in image_dicts:
        page_number = image_dict["page_number"]
        image_path = Path(image_dict["path"])
        if page_number in image_index:
            image_index[page_number].append(image_path)
        else:
            image_index[page_number] = [image_path]

    return image_index

# attach image metadata to the text nodes
def get_text_nodes(docs, json_dicts=None, image_dicts=None):
    """Split docs into nodes, by separator and attach image metadata to the text nodes"""
    nodes = []
    image_index = create_image_index(image_dicts) if image_dicts is not None else {}
    print("Image index: ", image_index)
    md_texts = [d["md"] for d in json_dicts] if (json_dicts is not None or json_dicts != {}) else None

    doc_chunks = [c for d in docs for c in d.text.split("---")]
    page_num = 0
    chunk_index = 0
    while chunk_index < len(doc_chunks):
        page_num += 1
        chunk_metadata = {"page_num": page_num, "image_paths": []}
        if image_index.get(page_num):
            for image in image_index[page_num]:
                chunk_metadata["image_paths"].append(str(image))
        if md_texts is not None:
            chunk_metadata["parsed_text_markdown"] = md_texts[chunk_index]
        chunk_metadata["parsed_text"] = doc_chunks[chunk_index]
        node = TextNode(text=doc_chunks[chunk_index], metadata=chunk_metadata)
        nodes.append(node)
        chunk_index += 1
    return nodes

class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.

    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: Ollama

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        # retrieve text nodes
        nodes = self.retriever.retrieve(query_str)
        # create ImageNode items from text nodes
        image_nodes = [
            NodeWithScore(node=ImageNode(image_path=image_path))
            for n in nodes for image_path in n.metadata.get("image_paths", [])
        ]

        # create context string from text nodes, dump into the prompt
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        image_docs = [image_node.node for image_node in image_nodes]
        # synthesize an answer from formatted text and images
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=image_docs
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
        )
    