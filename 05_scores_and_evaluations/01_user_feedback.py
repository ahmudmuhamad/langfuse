import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse, observe
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

# Step 1: LLM Assistant that captures and returns the trace_id
@observe(name="Ask Assistant")
def ask_assistant(question: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a concise programming tutor. Answer in 2 short bullet points."},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    
    reply = response.choices[0].message.content
    
    # Grab the current trace ID to return to the caller (e.g. your web API / frontend)
    current_trace_id = langfuse.get_current_trace_id()
    
    return {
        "reply": reply,
        "trace_id": current_trace_id
    }

# Step 2: Separate function simulating user feedback from the frontend
def submit_user_feedback(trace_id: str, rating_stars: int, feedback_comment: str):
    print(f"\n[User Feedback Event] Received feedback for Trace ID: {trace_id}")
    
    # 1. Attach a NUMERIC score (e.g., 1 to 5 stars)
    langfuse.create_score(
        trace_id=trace_id,
        name="star_rating",
        value=float(rating_stars),
        data_type="NUMERIC",
        comment=feedback_comment
    )
    
    # 2. Attach a CATEGORICAL score (e.g., thumbs_up / thumbs_down)
    langfuse.create_score(
        trace_id=trace_id,
        name="user_thumbs",
        value="thumbs_up" if rating_stars >= 4 else "thumbs_down",
        data_type="CATEGORICAL",
        comment="Auto-derived from star rating"
    )
    print("Feedback scores attached to trace!")

if __name__ == "__main__":
    print("1. User asks question...")
    result = ask_assistant("What is the difference between a tuple and a list in Python?")
    
    print("\nAssistant Answer:")
    print("--------------------------------------------------")
    print(result["reply"])
    print("--------------------------------------------------")
    print(f"Captured Trace ID: {result['trace_id']}")

    # 2. Simulate user reading the answer and submitting feedback a moment later
    submit_user_feedback(
        trace_id=result["trace_id"],
        rating_stars=5,
        feedback_comment="Crystal clear comparison, loved the brevity!"
    )

    # 3. Flush everything to Langfuse
    print("\nFlushing trace and scores to Langfuse...")
    langfuse.flush()
    print("Done! Open your Langfuse UI to inspect the Scores column and score badges.")
