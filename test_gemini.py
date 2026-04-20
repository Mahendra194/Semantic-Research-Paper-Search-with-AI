import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

try:
    key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello",
        config=types.GenerateContentConfig(temperature=0.3)
    )
    print("Success:", response.text)
except Exception as e:
    print("Exception thrown:", e)
