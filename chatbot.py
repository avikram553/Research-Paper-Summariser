from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",  # Model identifier on HuggingFace
    task="text-generation",  # Task type: generates text based on input
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # API token from .env file
)


chat_history = [
        SystemMessage(content= " Hello I am   Deep, Your AI assistant for today"),
]
print(chat_history[0].content)
model = ChatHuggingFace(llm=llm)

while True :   #Infine loop until user leaves

    user_input = input(" You : ")

    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit':

        break
    result=model.invoke(chat_history)
    print("Deep :", result.content)
    chat_history.append(AIMessage(content=result.content))
