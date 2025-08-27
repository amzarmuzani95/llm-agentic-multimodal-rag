import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent, UserInputRequestedEvent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig, SentenceTransformerEmbeddingFunctionConfig
# from autogen_agentchat.ui import Console

from autogen_ybocs_rag.indexer import SimpleDocumentIndexer
from system_prompt import BASIC_PROMPT

DOCS_PATH = r"data\documents"
MAX_TURNS = 20

# Initialize vector memory
rag_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="ocd_docs",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_ocd"),
        k=3,  # Return top k results
        score_threshold=0.3,  # Minimum similarity score
        embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default model for testing
            ),
    )
)

# Index OCD documentation
async def index_ocd_docs() -> None:
    indexer = SimpleDocumentIndexer(memory=rag_memory)
    sources = [os.path.join(DOCS_PATH, file) for file in os.listdir(DOCS_PATH) if file.endswith('.pdf')]
    print(sources)
    chunks: int = await indexer.index_documents(sources)
    print(f"Indexed {chunks} chunks from {len(sources)} documents in {DOCS_PATH} path")

# # define a "tool"
# def tool():
#     ...

# define team Config
def teamConfig():
    model = OllamaChatCompletionClient(
        model="llama3.2:1b"
    )

    # define user proxy to get feedback on responses. Maybe not necessary? Esp for literature review method
    user = UserProxyAgent("user_proxy", input_func=input)

    # define LLM to reply
    rag_agent = AssistantAgent(
        name="chatbot",
        system_message=(BASIC_PROMPT),
        model_client=model,
        memory=[rag_memory]
     )

    # define team for agents to work together
    team = RoundRobinGroupChat(
        participants=[rag_agent], #user
        max_turns=MAX_TURNS,
        termination_condition=TextMentionTermination(text="TERMINATE"),
    )

    return team

# co-routine for AI agents
async def orchestrate(team, task):
    # await rag_memory.clear()  # Clear existing memory
    # await index_ocd_docs()
    async for msg in team.run_stream(task=task):
        print("--" * 20)
        if isinstance(msg, TextMessage):
            print(message:=f"{msg.source}: {msg.content}")
            yield message
        elif isinstance(msg, ToolCallRequestEvent):
            print(message:=msg.to_text())
            yield message
        elif isinstance(msg, ToolCallExecutionEvent):
            print(message:=msg.to_text())
            yield message
        elif isinstance(msg, UserInputRequestedEvent):
            print(msg)
            print(message:=msg.to_text())
            yield message
    # await rag_memory.close() # close memory when done

# main co-routine
async def main(task):
    team = teamConfig()
    async for message in orchestrate(team, task):
        pass

if __name__ == "__main__":
    task = "Based on documents in your memory, what is Y-BOCS?" # example question
    asyncio.run(main(task))