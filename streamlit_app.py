import streamlit as st
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

# Configuración de la página
st.set_page_config(
    page_title="Asistente Legal IMMEX",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar variables de entorno
@st.cache_data
def load_environment():
    load_dotenv(find_dotenv())
    if "OPENAI_API_KEY" not in os.environ:
        st.error("❌ No se encuentra la API KEY de OpenAI")
        st.stop()
    return True

# Inicializar el agente (con cache para evitar recrearlo en cada interacción)
@st.cache_resource
def initialize_agent():
    """Inicializa el agente y sus herramientas"""
    try:
        # Modelo de chat
        chat_model = ChatOpenAI(model="gpt-4o-mini")
        
        # Herramientas
        web_search_tool = build_web_search_tool()
        sql_tools = build_sql_tool(llm=chat_model)
        rag_tool = build_enhanced_rag_tool(llm=chat_model)
        
        tools = [web_search_tool, rag_tool] + sql_tools
        
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Eres un agente experto que ayuda a resolver dudas sobre la ley aduanera y la "
             "Industria Manufacturera Maquiladora y de Servicios de Exportación IMMEX. "
             "Tienes acceso a herramientas como búsqueda SQL, búsqueda documental (RAG) y búsqueda web. "
             "Usa las herramientas cuando sea necesario y responde de forma clara y útil. "
             "Siempre cita las fuentes cuando uses información de documentos."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Crear agente
        agent = create_tool_calling_agent(llm=chat_model, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor, tools
        
    except Exception as e:
        st.error(f"❌ Error inicializando el agente: {str(e)}")
        st.stop()

def main():
    # Cargar configuración
    load_environment()
    
    # Header
    st.title("⚖️ Asistente Legal IMMEX")
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📋 Información del Sistema")
        
        st.markdown("""
        **Herramientas disponibles:**
        - 🔍 **Búsqueda Web**: Información actualizada
        - 📚 **RAG Documentos**: Leyes y reglamentos IMMEX
        - 🗄️ **Base de Datos**: Consultas SQL especializadas
        
        **Ejemplos de consultas:**
        - "¿Qué dice el artículo 15 sobre IMMEX?"
        - "Requisitos para importación temporal"
        - "Últimos cambios en la ley aduanera"
        """)
        
        if st.button("🗑️ Limpiar Conversación"):
            st.session_state.messages = []
            st.rerun()
    
    # Inicializar el agente
    with st.spinner("🔄 Inicializando sistema..."):
        agent_executor, tools = initialize_agent()
    
    st.success("✅ Sistema listo con 3 herramientas")
    
    # Inicializar el historial de conversación
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensaje de bienvenida
        welcome_msg = {
            "role": "assistant", 
            "content": "¡Hola! Soy tu asistente especializado en ley aduanera y IMMEX. ¿En qué puedo ayudarte hoy?"
        }
        st.session_state.messages.append(welcome_msg)
    
    # Mostrar historial de conversación
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre ley aduanera o IMMEX..."):
        
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analizando tu consulta..."):
                try:
                    # Crear contenedor para la respuesta
                    response_container = st.empty()
                    
                    # Ejecutar el agente
                    response = agent_executor.invoke({"input": prompt})
                    
                    # Mostrar la respuesta
                    assistant_response = response.get('output', 'No se pudo generar una respuesta.')
                    response_container.markdown(assistant_response)
                    
                    # Añadir al historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error procesando la consulta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

    # Footer con información adicional
    st.markdown("---")
    with st.expander("ℹ️ Información técnica"):
        st.markdown("""
        **Modelo**: GPT-4o-mini  
        **Herramientas**: RAG + SQL + Web Search  
        **Base de datos**: PostgreSQL con PGVector  
        **Embeddings**: OpenAI text-embedding-3-large  
        
        **Nota**: Este asistente está especializado en documentos legales mexicanos 
        relacionados con IMMEX y ley aduanera.
        """)

if __name__ == "__main__":
    main()