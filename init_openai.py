import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))