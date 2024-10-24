import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    loader = TextLoader("/Users/dylanthunn/Desktop/rag/mediumblog1.txt")
    document = loader.load()
    
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("Embedding...")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])    

    print("Done!")
