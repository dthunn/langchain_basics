from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


load_dotenv()


repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7)

template="""
Question:{question}
Answer:Lets think step by step.
"""

prompt = PromptTemplate(template=template,input_variables=["question"])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question":"What is the capital of France?"})

print(response)

# model_name = "BAAI/bge-small-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )
# embedding = hf.embed_query("hi this is harrison")

