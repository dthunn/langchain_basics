from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert AI Engineer. Provide me answers based on the questions"),
        ("user","{input}")
    ]

)

chain = prompt | llm | StrOutputParser()

response=chain.invoke({"input":"Can you tell me about Langsmith?"})
print(response)


