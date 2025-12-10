import os,getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your openai API key: ")
#os.environ["LANGSMITH_TRACING"] = "true"

from langchain_community.embeddings import FakeEmbeddings
embeddings = FakeEmbeddings(size=50)
vector = embeddings.embed_query("hello, world!")
vector[:5]
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)
document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

documents = [
    document_1,
    document_2,
    document_3,
   ]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke("langchain", filter={"source": "news"})
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    query:str
    context:str

# Step 3: Define model node
from langchain.messages import SystemMessage

#from langgraph.nodes import LLMNode
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain.chat_models import init_chat_model

#@tool
def retriever_node(state: dict) -> str:
 response=retriever.invoke(query)
 print('retriver node response:',response)
 context=response[0]
 return {"context":context}
# 2. Define nodes

#tools = [retriever_node]
from langchain.messages import HumanMessage


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    print('llm call with state',query)
    return {
        "messages": [
            llm.invoke(
                [HumanMessage(content=query),
                    SystemMessage(
                        content="You are a helpful assistant. Use the following context in your response:{context}"
                    )
                ]
                + state["messages"]
            )
        ],
      "llm_calls": state.get('llm_calls', 0) + 1
    }
# 3. Build graph
#graph = Graph()
agent_builder  = StateGraph(MessagesState)
agent_builder .add_node("llm_call", llm_call)
agent_builder .add_node("tool_node", retriever_node)

# Connect retriever → generator

# Add edges to connect nodes
agent_builder .add_edge(START, "tool_node")#retriver
agent_builder .add_edge("tool_node", "llm_call")#generator
agent_builder .add_edge("llm_call",END)
agent = agent_builder.compile()
# 4. Run workflow
query = "What is langchain?"
from langchain.messages import HumanMessage
messages = [HumanMessage(content=query)]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
