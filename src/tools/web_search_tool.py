import os
from langchain_tavily import TavilySearch

def build_web_search_tool():
    # Asegúrate de que la variable de entorno esté cargada
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Falta la API Key de Tavily. Define TAVILY_API_KEY en el entorno.")

    return TavilySearch(max_results=3)