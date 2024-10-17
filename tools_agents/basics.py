from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import openai


load_dotenv()

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)


api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vector_db = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector_db.as_retriever()

retriever_tool = create_retriever_tool(retriever, "langsmith-search", "Search any information about Langsmith ")

tools = [wiki,arxiv,retriever_tool]

llm=ChatGroq(model_name="Llama3-8b-8192")

prompt = hub.pull("hwchase17/openai-functions-agent")

agent=create_openai_tools_agent(llm, tools, prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

# result = agent_executor.invoke({"input":" Tell me about Langsmith"})

result = agent_executor.invoke({"input":"What is machine learning"})

print(result)

