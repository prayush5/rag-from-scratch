import json
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query_type: str = Field(..., description="Type of query to perform. E.g. 'sales_by_date', 'table_availability'")
    date_or_time: str = Field(..., description="Target date for reservation. Format 'YYYY-MM-DD'")

def query_database(query_type: str, date_or_time: str):
    if query_type == "table_availability":
        return json.dumps({"date": date_or_time, "available_slots": "[18:00, 19:00, 20:00, 21:00]"})
    elif query_type == "menu_item":
        return json.dumps({"item": date_or_time, "price": 18.99})
    return json.dumps({"error": f"Unknown query type: {query_type}"})

DB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Queries the database for table availability or menu item prices.",
        "parameters": DatabaseQueryInput.model_json_schema()
    }
}