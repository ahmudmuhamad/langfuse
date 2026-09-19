from langfuse import Langfuse, observe
from dotenv import load_dotenv


load_dotenv()
langfuse = Langfuse()




@observe(name = "Fetch senstive data", capture_output = False, as_type = "tool")
def fetch_user_secret(user_id: str):
    return {
        "api_key": "secret_abc123", 
        "pin": 9988
    }

@observe(name= "User Profile Agent", as_type = "agent")
def user_agent_workflow(user_id: str):
    fetch_user_secret(user_id)
    return f"Successfully authenticated user {user_id}"

if __name__ == "__main__":
    user_agent_workflow("user_42")
    langfuse.flush()