import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

def setup_llm(vectorstore, llm_name="groq"):
    """Sets up the Language Model."""

    if llm_name == "groq":
        llm = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=groq_api_key,
        )
    
    elif llm_name == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
    
    else:
        raise ValueError("Invalid LLM. Please select 'groq' or 'gemini'.")
    
    # Set up RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa