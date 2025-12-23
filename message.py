'''#Langchain has already identified the problem that we donot have the role base chat history and it has already
#resolved using messages.
There aren three types of messages in Langchain which we will use in this file 
'''

from langchain_core.messages import SystemMessage , HumanMessage , AIMessage

from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace

from dotenv import load_dotenv

import os

load_dotenv()


#Initialising a model()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",  # Model identifier on HuggingFace
    task="text-generation",  # Task type: generates text based on input
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # API token from .env file
)

model=ChatHuggingFace(llm=llm)

messsages= [

        SystemMessage(content=" You are an amazing assistant. Your name is Deep."),
        HumanMessage(content = "Tell me something about you ")
]

result=model.invoke(messsages)

messsages.append(result)

print(messsages)

