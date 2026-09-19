import os
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import OpenAI

# 1. Load environment variables
load_dotenv()

# 2. Instantiate clients
langfuse = Langfuse()

# Groq uses the OpenAI API standard, so Langfuse's OpenAI wrapper works directly!
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

@observe()
def generate_poem(topic: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a creative poet. Write a concise 2-line poem."},
            {"role": "user", "content": f"Write a poem about {topic}."}
        ],
        temperature=0.7
    )
    poem = response.choices[0].message.content
    return poem

if __name__ == "__main__":
    print("Generating poem via Groq (with Langfuse tracing)...\n")
    result = generate_poem(topic="artificial intelligence")
    
    print("Generated Poem:")
    print("----------------")
    print(result)
    print("----------------")

    print("\nFlushing events to Langfuse...")
    langfuse.flush()
    print("Done! Check your Langfuse dashboard to see the Generation details and tokens.")
