from dotenv import load_dotenv
import os
load_dotenv()

MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN")
print(MINERU_API_TOKEN)