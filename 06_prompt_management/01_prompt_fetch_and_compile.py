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

PROMPT_NAME = "onboarding-assistant"

def ensure_prompt_exists():
    """
    Checks if the prompt exists in Langfuse. If not, creates version 1.
    In real production, you or your team would create and edit this directly in the Langfuse UI!
    """
    try:
        prompt = langfuse.get_prompt(PROMPT_NAME, type="chat")
        print(f"Loaded existing prompt '{PROMPT_NAME}' (Version {prompt.version}) from Langfuse.")
        return prompt
    except Exception:
        print(f"Prompt '{PROMPT_NAME}' not found. Creating it in Langfuse...")
        prompt = langfuse.create_prompt(
            name=PROMPT_NAME,
            type="chat",
            prompt=[
                {
                    "role": "system",
                    "content": (
                        "You are an onboarding guide for {{product_name}}.\n"
                        "Welcome {{user_name}} who is on a {{device_type}}.\n"
                        "Maintain a {{tone}} tone. Answer in 2 crisp sentences."
                    )
                },
                {
                    "role": "user",
                    "content": "My goal today is: {{user_goal}}"
                }
            ],
            # We can bundle recommended model settings inside the prompt itself!
            config={
                "model": "openai/gpt-oss-120b",
                "temperature": 0.4
            },
            labels=["production"],
            commit_message="Initial v1 release of onboarding prompt"
        )
        print(f"Created prompt '{PROMPT_NAME}' (Version {prompt.version}) with 'production' label!")
        return prompt

@observe(name="Onboarding Workflow")
def run_onboarding_turn(user_name: str, device_type: str, user_goal: str) -> str:
    # 1. Fetch the active prompt (Langfuse caches this locally so it's super fast)
    prompt = langfuse.get_prompt(PROMPT_NAME, type="chat")

    # 2. Compile variables into formatted OpenAI chat messages
    compiled_messages = prompt.compile(
        product_name="CloudSync Studio",
        user_name=user_name,
        device_type=device_type,
        tone="enthusiastic and helpful",
        user_goal=user_goal
    )

    # 3. Read model parameters directly from the prompt configuration
    model_name = prompt.config.get("model", "openai/gpt-oss-120b")
    temperature = prompt.config.get("temperature", 0.5)

    # 4. Link the prompt version to all generations executed in this context!
    with propagate_attributes(prompt=prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=compiled_messages,
            temperature=temperature
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Ensure prompt is registered in Langfuse
    active_prompt = ensure_prompt_exists()
    print(f"Using Prompt: '{active_prompt.name}' | Version: {active_prompt.version} | Labels: {active_prompt.labels}\n")

    print("Running onboarding workflow with dynamically compiled prompt...")
    answer = run_onboarding_turn(
        user_name="Marcus",
        device_type="iPad Pro",
        user_goal="Set up automated photo backups to my private cloud"
    )

    print("\nAssistant Output:")
    print("--------------------------------------------------")
    print(answer)
    print("--------------------------------------------------")

    print("\nFlushing events to Langfuse...")
    langfuse.flush()
    print("Done! Check your Langfuse UI under 'Prompts' and 'Traces'.")
