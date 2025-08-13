import os
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from src.tools.web_search_tool import build_web_search_tool
from src.tools.sql_tool import build_sql_tool
from src.tools.rag_tool import build_enhanced_rag_tool

_ = load_dotenv(find_dotenv())
if "OPENAI_API_KEY" not in os.environ:
    print("No se encuentra la API KEY del servicio LLM")
    sys.exit(1)
    
chat_model = ChatOpenAI(model="gpt-4o-mini")

web_search_tool = build_web_search_tool()
sql_tools = build_sql_tool(llm=chat_model)
rag_tool = build_enhanced_rag_tool(llm=chat_model)
    
tools = [web_search_tool, rag_tool] + sql_tools

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un agente experto que ayuda a resolver dudas sobre la ley aduanera y la Industria Manufacturera Maquiladora y de Servicios de Exportación IMMEX. "
     "Tienes acceso a herramientas como búsqueda SQL, búsqueda documental (RAG) y búsqueda web. "
     "Usa las herramientas cuando sea necesario y responde de forma clara y útil."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=chat_model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Chat interactivo
print("=== Chat Interactivo ===")
print("Escribe 'salir' para terminar la conversación\n")

while True:
    try:
        # Solicitar input del usuario
        user_input = input("Tu pregunta: ").strip()
        
        # Verificar si quiere salir
        if user_input.lower() == "salir":
            print("¡Hasta luego!")
            break
        
        # Verificar que no esté vacío
        if not user_input:
            print("Por favor, escribe una pregunta válida.\n")
            continue
        
        # Procesar la pregunta
        print("\nProcesando...\n")
        response = agent_executor.invoke({"input": user_input})
        
        # Mostrar la respuesta
        print(f"\nRespuesta: {response['output']}\n")
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print("Intenta de nuevo.\n")