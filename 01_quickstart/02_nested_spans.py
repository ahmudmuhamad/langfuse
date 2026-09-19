from langfuse import observe, Langfuse
from dotenv import load_dotenv 


load_dotenv()
langfuse = Langfuse()



@observe()
def calculate_discount(price: float, discount_code: str):
    if discount_code == "SAVE10":
        price = 0.9 * price 
        return price
    else:
        return price

@observe()
def checkout(item: str, price: float, discount_code: str):
    final_price = calculate_discount(price, discount_code)
    return {
        "item": item, 
        "original_price": price, 
        "final_price": final_price,
        "status": "completed"

    }

if __name__ == "__main__":
    checkout("Lenovo Legion Laptop", price=100.0, discount_code="SAVE10")
    langfuse.flush()