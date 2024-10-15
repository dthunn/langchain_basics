from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

loader = TextLoader("speech.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
docs = text_splitter.split_documents(documents)

embeddings=OpenAIEmbeddings()
db=FAISS.from_documents(docs,embeddings)

query="How does the speaker describe the desired outcome of the war?"
docs=db.similarity_search(query)
# print(docs[0].page_content)

retriever=db.as_retriever()
docs=retriever.invoke(query)
print(docs[0].page_content)

db.save_local("faiss_index")

new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
docs=new_db.similarity_search(query)

print(docs)