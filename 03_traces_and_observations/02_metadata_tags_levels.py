import os
import sys
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

# Step 1: Span with dynamic Warning Level and Status Message
@observe(name="Analyze & Route Query")
def analyze_query(query: str) -> dict:
    word_count = len(query.split())
    
    # If the query is unusually long, flag a WARNING on this span
    if word_count > 25:
        langfuse.update_current_span(
            level="WARNING",
            status_message=f"Query is unusually long ({word_count} words). High token consumption expected."
        )
    else:
        langfuse.update_current_span(
            level="DEFAULT",
            status_message="Query length is within normal budget."
        )
        
    return {
        "word_count": word_count,
        "is_complex": word_count > 25
    }

# Step 2: Generation
@observe(name="Generate Concise Answer", as_type="generation")
def generate_reply(query: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a concise tech assistant. Answer in one crisp sentence."},
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Root Workflow: Applies Tags and Metadata to the entire execution context
@observe(name="Enriched Support Workflow")
def run_workflow(query: str) -> dict:
    # propagate_attributes attaches tags and metadata to all current and child spans
    with propagate_attributes(
        tags=["production", "tier-enterprise", "mobile-app"],
        metadata={
            "app_version": "4.12.0",
            "device": "iPhone 16 Pro",
            "region": "eu-central-1"
        }
    ):
        analysis = analyze_query(query)
        reply = generate_reply(query)
        
        return {
            "analysis": analysis,
            "reply": reply
        }

if __name__ == "__main__":
    print("Running workflow with custom Tags, Metadata, and Warning Level...\n")
    
    # Intentionally long query to trigger the WARNING level in analyze_query
    long_query = (
        "Could you explain in comprehensive technical detail how distributed database "
        "replication operates when network partitions occur, specifically analyzing the trade-offs "
        "outlined in the CAP theorem regarding consistency and partition tolerance?"
    )
    
    result = run_workflow(long_query)
    
    print("Execution Output:")
    print("-----------------")
    print(f"Query Analysis: {result['analysis']}")
    print(f"Reply: {result['reply']}")
    print("-----------------")

    print("\nFlushing events to Langfuse...")
    langfuse.flush()
    print("Done! Check your Langfuse UI to see the Tags, Metadata, and WARNING status.")
