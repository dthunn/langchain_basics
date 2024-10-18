from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(model="Gemma2-9b-It")

# messages = [
#     SystemMessage(content="Translate the following from English to French"),
#     HumanMessage(content="Hello How are you?")
# ]

# result = model.invoke(messages)

parser = StrOutputParser()

generic_template = "Translate the following into {language}:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

chain = prompt | model | parser

result = chain.invoke({"language":"French","text":"Hello"})

print(result)