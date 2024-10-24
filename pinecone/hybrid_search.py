import os
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder


load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "hybrid-search-langchain-pinecone"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index=pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

bm25_encoder=BM25Encoder().default()

sentences=[
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

]

## tfidf values on these sentence
bm25_encoder.fit(sentences)

## store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)

retriever.add_texts(
    [
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

    ]
)

print(retriever.invoke("What city did i visit first"))
