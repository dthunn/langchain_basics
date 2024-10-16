from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


load_dotenv() 

model = ChatGroq(model="Gemma2-9b-It")

# model.invoke(
#     [
#         HumanMessage(content="Hi , My name is Dylan and I am a Software Engineer"),
#         AIMessage(content="Hello Dylan! It's nice to meet you. \n\nAs a Software Engineer, what kind of projects are you working on these days? \n\nI'm always eager to learn more about the exciting work being done in the field of AI.\n"),
#         HumanMessage(content="Hey What's my name and what do I do?")
#     ]
# )

store={}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config={"configurable":{"session_id":"chat1"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi , My name is Dylan and I am a Software Engineer")],
#     config=config
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config,
# )

# config1 ={ "configurable":{"session_id":"chat2"}}
# response = with_message_history.invoke(
#     [HumanMessage(content="Whats my name")],
#     config=config1
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="Hey My name is Luffy")],
#     config=config1
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="Whats my name")],
#     config=config1
# )

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful assistant. Answer all the question to the nest of your ability"),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# )

# chain= prompt | model

# response = chain.invoke({"messages":[HumanMessage(content="Hi My name is Dylan")]})

# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# config = {"configurable": {"session_id": "chat3"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi My name is Dylan")],
#     config=config
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config,
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# response = chain.invoke({"messages":[HumanMessage(content="Hi My name is Dylan")], "language": "English"})

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable": {"session_id": "chat4"}}
response = with_message_history.invoke(
    {'messages': [HumanMessage(content="Hi,I am Dylan")], "language":"English"},
    config=config
)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "English"},
    config=config,
)

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

chain=(
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
    
)

response = chain.invoke(
    {
    "messages": messages + [HumanMessage(content="What ice cream do I like")],
    "language": "English"
    }
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config={"configurable":{"session_id":"chat5"}}

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

print(response)






