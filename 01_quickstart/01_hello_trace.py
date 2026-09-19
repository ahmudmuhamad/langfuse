from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse import observe



load_dotenv()

langfuse = Langfuse()
print("Auth Check", langfuse.auth_check())


@observe()
def process_order(item: str, quantity: int):
    price_per_item = 25.0
    total = price_per_item * quantity
    return {
        "status": "confirmed",
        "item": item,
        "quantity": quantity,
        "total": total
    }

if __name__ == "__main__":
    # Call the observed function
    response = process_order(item="Wireless Headphones", quantity=2)
    print("Function returned:", response)

    # Flush all events before the script exits
    langfuse.flush()
    print("Trace sent successfully!")