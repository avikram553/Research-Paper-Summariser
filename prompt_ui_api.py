# Import necessary libraries
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint  # LangChain integration for HuggingFace models
from dotenv import load_dotenv  # Load environment variables from .env file
import streamlit as st  # Streamlit for creating web UI
import os  # Operating system interface for accessing environment variables
from langchain_core.prompts import PromptTemplate , load_prompt
# Load environment variables from .env file (contains API keys)
load_dotenv()

###############
# Model Configuration
###############

# Initialize HuggingFace Endpoint with DeepSeek model
# This connects to the HuggingFace API to use the DeepSeek-V3.2 model
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",  # Model identifier on HuggingFace
    task="text-generation",  # Task type: generates text based on input
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # API token from .env file
)

# Wrap the endpoint in ChatHuggingFace for conversational interface
# This enables chat-style interactions with the model
model=ChatHuggingFace(llm=llm)

# Alternative: Direct model invocation (currently commented out)
#model=llm()

###############
# Streamlit UI
###############

# Display the main header
st.header("Research Summariser")


paper_input=st.selectbox("Select any paper" , [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Language Models are Few-Shot Learners",
    "Deep Contextualized Word Representations",
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
])


style_input=st.selectbox("Select Explanation Style" ,[
    "Beginner-Friendly",
    "Example-Based",
    "Technical/Academic",
    "Code-Focused",
    "Visual/Diagram-Based",
    "Bullet Points",
    "ELI5 (Explain Like I'm 5)",
    "Step-by-Step",
    "Comparison-Based",
    "Real-World Applications"
])

length_input=st.selectbox("Length Input", [
    "Brief",
    "Short",
    "Medium",
    "Detailed",
    "Comprehensive"
])



template=load_prompt('template.json')

prompt=template.invoke({'paper_input': paper_input,
                 'style_input': style_input,
                 'length_input': length_input})
# Create a text input field for user to enter their prompt/research text
#user_input= st.text_input("Enter your prompt")

# Create a button that triggers the summarization
if st.button("Summarise"):
    # When button is clicked, invoke the model with user's input
    result=model.invoke(prompt)
    # Display the generated summary from the model
    st.write(result.content)




