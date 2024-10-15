from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# embeddings=(
#     OllamaEmbeddings(model="gemma:2b")  ##by default it ues llama2
# )

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# loader=TextLoader('speech.txt')
# docs=loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# final_documents = text_splitter.split_documents(docs)

# db=Chroma.from_documents(final_documents, embeddings)

# query = "It will be all the easier for us to conduct ourselves as belligerents"
# retrieved_results = db.similarity_search(query)

text = "this is latest documents"
query_result = embeddings.embed_query(text)

print(query_result)

