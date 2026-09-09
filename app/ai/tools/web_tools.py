import json
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web (e.g. 'restaurant parking options nearby').")

def search_web(query: str):
    #perform a web search using duckduckgo
    try:
        results = list(DDGS().text(query, max_results=3))
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})
    
WEB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Performs a web search using DuckDuckGo and returns the top search results.",
        "parameters": WebSearchInput.model_json_schema()
    }
}
        