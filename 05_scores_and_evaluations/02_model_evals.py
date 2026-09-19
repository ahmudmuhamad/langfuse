import json
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

# 2. Instantiate clients
langfuse = Langfuse()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Step 1: The primary application LLM call
@observe(name="Primary Assistant", as_type="generation")
def generate_answer(question: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a concise engineering assistant. Answer in 2-3 sentences."},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Step 2: The LLM-as-a-Judge Evaluator
# Notice: as_type="evaluator" is an official Langfuse observation type!
@observe(name="Conciseness Evaluator", as_type="evaluator")
def evaluate_conciseness(question: str, answer: str) -> dict:
    judge_prompt = f"""You are an expert AI evaluator.
Evaluate the following assistant answer to the user question on a scale of 0.0 to 1.0 based on CONCISENESS and RELEVANCE.
- 1.0: Perfectly concise, directly answered, no fluff.
- 0.5: Acceptable, but contained unnecessary explanation.
- 0.0: Rambling, repetitive, or failed to answer.

User Question: "{question}"
Assistant Answer: "{answer}"

Output ONLY a JSON object with this exact format:
{{"score": 0.95, "reasoning": "Direct explanation with zero wasted words."}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an automated evaluation judge. Respond strictly in valid JSON."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.1
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    # Clean possible markdown code fences if returned by the LLM
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`").replace("json", "").strip()

    eval_data = json.loads(raw_content)

    # Attach the evaluated score directly to the current trace!
    langfuse.score_current_trace(
        name="conciseness_eval",
        value=float(eval_data["score"]),
        data_type="NUMERIC",
        comment=eval_data["reasoning"]
    )
    
    return eval_data

# Root Orchestrator: Runs the task and immediately runs the automated eval
@observe(name="Automated Eval Pipeline")
def run_pipeline(question: str) -> dict:
    # 1. Generate the answer
    answer = generate_answer(question)
    
    # 2. Automatically evaluate the answer using LLM-as-a-judge
    evaluation = evaluate_conciseness(question, answer)
    
    return {
        "question": question,
        "answer": answer,
        "evaluation": evaluation
    }

if __name__ == "__main__":
    test_question = "What is Docker and why is it useful?"
    print(f"Running pipeline with automated LLM judge for: '{test_question}'\n")

    result = run_pipeline(test_question)

    print("Assistant Answer:")
    print("--------------------------------------------------")
    print(result["answer"])
    print("--------------------------------------------------")

    print("\nAutomated Judge Evaluation:")
    print("--------------------------------------------------")
    print(f"Score:     {result['evaluation']['score']} / 1.0")
    print(f"Reasoning: {result['evaluation']['reasoning']}")
    print("--------------------------------------------------")

    print("\nFlushing trace and evaluation score to Langfuse...")
    langfuse.flush()
    print("Done! Inspect the trace tree and notice the 'Evaluator' span and attached score badge.")
