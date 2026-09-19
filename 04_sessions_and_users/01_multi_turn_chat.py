import os
import sys
import time
from dotenv import load_dotenv
from langfuse import Langfuse, observe, propagate_attributes
from langfuse.openai import OpenAI

# Fix Windows console UTF-8 printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load environment variables
load_dotenv()

# 2. Instantiate Langfuse and Groq client
langfuse = Langfuse()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Each individual turn in the conversation is an observed Trace
@observe(name="Chat Turn")
def chat_turn(conversation_messages: list, new_user_message: str) -> str:
    messages_payload = conversation_messages + [{"role": "user", "content": new_user_message}]
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages_payload,
        temperature=0.6
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Define consistent user and session identifiers
    USER_ID = "user_john_doe"
    # Using a timestamped session ID so each run gets its own fresh thread in the UI
    SESSION_ID = f"session_travel_{int(time.time())}"

    print(f"Starting multi-turn chat session:")
    print(f" User ID:    {USER_ID}")
    print(f" Session ID: {SESSION_ID}\n")

    # Conversation state maintained across turns
    history = [
        {
            "role": "system",
            "content": "You are a concise travel concierge. Keep each reply under 3 sentences."
        }
    ]

    # The 3-turn user dialogue
    turns = [
        "I want to visit a sunny European country known for incredible food and history.",
        "What are the top 2 cities to visit there?",
        "Which of those two has better beaches?"
    ]

    for i, user_message in enumerate(turns, start=1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_message}")

        # propagate_attributes stamps USER_ID and SESSION_ID onto this turn's trace
        with propagate_attributes(
            user_id=USER_ID,
            session_id=SESSION_ID,
            tags=["travel-assistant", "demo"]
        ):
            assistant_reply = chat_turn(history, user_message)

        # Update conversation history for the next turn
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})

        print(f"Assistant: {assistant_reply}\n")

    # Flush all turns to Langfuse
    print("Flushing all 3 chat turns to Langfuse...")
    langfuse.flush()
    print("Done! Open your Langfuse UI to view the conversation thread under 'Sessions'.")
