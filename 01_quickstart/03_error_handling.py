from dotenv import load_dotenv
from langfuse import Langfuse, observe

load_dotenv()

langfuse = Langfuse()

@observe()
def apply_coupon(coupon: str) -> float:
    if coupon == "VALID10":
        return 0.10
    elif coupon == "EXPIRED":
        raise ValueError("Coupon code 'EXPIRED' has expired!")
    else:
        raise ValueError(f"Unknown coupon code: '{coupon}'")

@observe()
def process_payment(amount: float, coupon: str) -> dict:
    # This call will raise an exception inside the child span
    discount = apply_coupon(coupon)
    final_amount = amount * (1.0 - discount)
    return {
        "status": "success",
        "amount_paid": final_amount
    }

if __name__ == "__main__":
    print("Executing payment flow with an invalid coupon...")
    
    try:
        process_payment(amount=100.0, coupon="EXPIRED")
    except ValueError as e:
        print(f"Caught error in main application: {e}")
    finally:
        # Always flush, even if an error occurred!
        print("\nFlushing trace to Langfuse...")
        langfuse.flush()
        print("Done! Check your Langfuse dashboard.")
