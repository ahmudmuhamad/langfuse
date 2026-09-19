import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.experiment import Evaluation
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

DATASET_NAME = "geography-capitals-benchmark"

def ensure_dataset():
    """Create the benchmark dataset with test items if it does not already exist."""
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
        print(f"Loaded existing dataset '{DATASET_NAME}' with {len(dataset.items)} items.")
        return dataset
    except Exception:
        print(f"Dataset '{DATASET_NAME}' not found. Creating new benchmark dataset...")
        dataset = langfuse.create_dataset(
            name=DATASET_NAME,
            description="Benchmark dataset testing geography capital city retrieval accuracy"
        )
        
        # Benchmark ground-truth items
        test_items = [
            {"input": "What is the capital of France?", "expected_output": "Paris"},
            {"input": "What is the capital of Japan?", "expected_output": "Tokyo"},
            {"input": "What is the capital of Australia?", "expected_output": "Canberra"},
            {"input": "What is the capital of Canada?", "expected_output": "Ottawa"},
            {"input": "What is the capital of Brazil?", "expected_output": "Brasilia"}
        ]
        
        for item in test_items:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input=item["input"],
                expected_output=item["expected_output"]
            )
            
        dataset = langfuse.get_dataset(DATASET_NAME)
        print(f"Created dataset with {len(dataset.items)} benchmark items!")
        return dataset

# 3. The Task function to evaluate
# Must accept 'item' as a keyword argument
def geography_task(*, item, **kwargs) -> str:
    question = item.input
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Answer the question with JUST the name of the capital city, nothing else."},
            {"role": "user", "content": question}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 4. The Evaluator function
# Compares the model output against expected ground truth
def accuracy_evaluator(*, input, output, expected_output=None, **kwargs) -> Evaluation:
    if not expected_output:
        return Evaluation(name="accuracy", value=0.0, comment="No expected output provided")

    is_correct = expected_output.strip().lower() in output.strip().lower()
    return Evaluation(
        name="accuracy",
        value=1.0 if is_correct else 0.0,
        comment=f"Answer '{output}' matches expected '{expected_output}'" if is_correct else f"Mismatch: got '{output}', expected '{expected_output}'"
    )

if __name__ == "__main__":
    # Step 1: Ensure dataset is loaded
    dataset = ensure_dataset()

    # Step 2: Run the experiment across all items concurrently
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name = f"gpt-oss-120b-run-{run_timestamp}"

    print(f"\nLaunching Langfuse Experiment Run: '{experiment_name}'...")
    print(f"Evaluating {len(dataset.items)} items with automated accuracy scoring...\n")

    experiment_result = langfuse.run_experiment(
        name=experiment_name,
        data=dataset.items,
        task=geography_task,
        evaluators=[accuracy_evaluator],
        max_concurrency=5
    )

    print("Experiment run completed!")
    print(f"Run Name: {experiment_result.run_name}")

    # Step 3: Flush traces to Langfuse
    print("\nFlushing experiment results to Langfuse...")
    langfuse.flush()
    print("Done! Check your Langfuse UI under 'Datasets' to view the benchmark results table.")
