from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain



load_dotenv()

loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents,embeddings)

## Query From a vector db
query = "LangSmith has two usage limits: total traces and extended"
result = vector_store.similarity_search(query)

llm = ChatOpenAI()

prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

response=retrieval_chain.invoke({"input":"LangSmith has two usage limits: total traces and extended"})
print(response)

