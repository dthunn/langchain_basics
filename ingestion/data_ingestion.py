from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
import json
import requests

load_dotenv()

loader = TextLoader("speech.txt")
text_documents = loader.load()

loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",))
web_documents = loader.load()

docs = ArxivLoader(query="1706.03762", load_max_docs=2).load()
print(docs)

docs = WikipediaLoader(query="Generative AI", load_max_docs=2).load()
print(docs)

pdf_loader = PyPDFLoader("attention.pdf")
pdf_documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = text_splitter.split_documents(pdf_documents)

print(final_documents)

speech=""
with open("speech.txt") as f:
    speech=f.read()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_splitter.create_documents([speech])
print(text)

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_json(json_data)

for chunk in json_chunks[:3]:
    print(chunk)
