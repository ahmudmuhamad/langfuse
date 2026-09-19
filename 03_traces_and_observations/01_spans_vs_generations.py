import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import OpenAI

# Fix Windows console UTF-8 printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load environment variables
load_dotenv()

# 2. Instantiate Langfuse and OpenAI-compatible Groq client
langfuse = Langfuse()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Step 1: Normal Span (DB lookup / Preprocessing)
@observe(name="Fetch User Profile", as_type="span")
def fetch_user_profile(user_id: str) -> dict:
    # Simulating a database lookup
    mock_db = {
        "user_101": {"name": "Alice", "plan": "Enterprise", "tone": "friendly and professional"},
        "user_102": {"name": "Bob", "plan": "Free", "tone": "concise and direct"}
    }
    return mock_db.get(user_id, {"name": "Guest", "plan": "Standard", "tone": "helpful"})

# Step 2: Generation (LLM Call)
@observe(name="Generate LLM Response", as_type="generation")
def generate_support_reply(query: str, profile: dict) -> str:
    system_prompt = (
        f"You are a customer support agent. The user's name is {profile['name']} "
        f"on the {profile['plan']} plan. Match your response tone to be {profile['tone']}."
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Step 3: Normal Span (Sanitization / Postprocessing)
@observe(name="Format & Sanitize Output", as_type="span")
def format_output(raw_reply: str, plan: str) -> dict:
    footer = f"\n\n[Priority Support Enabled]" if plan == "Enterprise" else "\n\n[Standard Support]"
    return {
        "formatted_text": raw_reply.strip() + footer,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }

# Root Orchestrator: Combines Spans and Generations into one cohesive trace
@observe(name="Customer Support Pipeline")
def support_pipeline(user_id: str, query: str) -> dict:
    # Step 1: Run Span
    profile = fetch_user_profile(user_id)
    
    # Step 2: Run Generation
    reply = generate_support_reply(query, profile)
    
    # Step 3: Run Span
    final_output = format_output(reply, profile["plan"])
    
    return {
        "user_id": user_id,
        "user_name": profile["name"],
        "plan": profile["plan"],
        "response": final_output["formatted_text"],
        "timestamp": final_output["processed_at"]
    }

if __name__ == "__main__":
    print("Running Customer Support Pipeline...")
    result = support_pipeline(
        user_id="user_101",
        query="How do I configure custom webhook alerts for my account?"
    )
    
    print("\nFinal Pipeline Result:")
    print("--------------------------------------------------")
    print(f"User: {result['user_name']} ({result['plan']})")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Response:\n{result['response']}")
    print("--------------------------------------------------")

    print("\nFlushing events to Langfuse...")
    langfuse.flush()
    print("Done! Check your Langfuse UI to inspect the Spans and Generation in the tree.")