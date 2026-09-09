import json
from pydantic import BaseModel, Field

class CalculateBillInput(BaseModel):
    subtotal: float = Field(..., description="Subtotal bill amount in NPR before tax.")
    tax_rate: int = Field(default=13, description="Tax percentage (default 13%)")
    split_into_n: int = Field(default=1, description="Number of people splitting the bill")

def calculate_bill(subtotal: float, tax_rate: int = 13, split: int = 1):
    total_tax = subtotal * (tax_rate / 100)
    grand_total = subtotal + total_tax
    per_person_amount = grand_total / split

    return json.dumps({
        "subtotal": subtotal,
        "tax": total_tax,
        "grand_total": round(grand_total, 2),
        "per_person": round(per_person_amount, 2)
    })

BILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_bill",
        "description": "Calculates total restaurant bill including tax and split per person accurately.",
        "parameters": CalculateBillInput.model_json_schema()
    }
}