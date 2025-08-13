from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

USER = "postgres"
PASSWORD = "contrasena"
HOST = "localhost"
PORT = "5432"
DB_NAME = "tools_agent_app"
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"

# Usar la nueva colección optimizada para documentos legales
COLLECTION_NAME = "legal_documents_v3"

class EnhancedLegalRetriever:
    """Retriever especializado para documentos legales con búsqueda híbrida"""
    
    def __init__(self, vectorstore, k=5):
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query: str):
        """Búsqueda híbrida que combina similitud semántica y filtrado por metadata"""
        
        # Detectar si se busca un artículo específico
        import re
        article_pattern = re.search(r'art[íi]culo\s+(\d+)', query.lower())
        
        if article_pattern:
            article_num = int(article_pattern.group(1))
            # Buscar específicamente por número de artículo
            results = self.vectorstore.similarity_search(
                query, 
                k=self.k,
                filter={"article_number": article_num}
            )
            
            # Si no encuentra por filtro, buscar normalmente pero priorizar artículos
            if not results:
                all_results = self.vectorstore.similarity_search(query, k=self.k * 2)
                results = [doc for doc in all_results if doc.metadata.get('article_number') == article_num]
                if not results:
                    results = all_results[:self.k]
        else:
            # Búsqueda normal con mayor número de resultados
            results = self.vectorstore.similarity_search(query, k=self.k)
        
        return results

def build_enhanced_rag_tool(llm: BaseLanguageModel) -> Tool:
    """Construye una herramienta RAG mejorada para documentos legales"""
    
    # Inicializar vector store con embeddings mejorados
    vectorstore = PGVector(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # Usar retriever personalizado
    retriever = EnhancedLegalRetriever(vectorstore, k=4)
    
    # Template mejorado para documentos legales
    legal_prompt_template = """Eres un asistente especializado en documentos legales mexicanos. 
Usa el contexto proporcionado para responder la pregunta de manera precisa y detallada.

IMPORTANTE:
- Si encuentras información sobre artículos específicos, SIEMPRE menciona el número del artículo
- Si encuentras fracciones, incisos o párrafos, menciónalos específicamente
- Proporciona citas exactas cuando sea posible
- Si no tienes información suficiente, dilo claramente

Contexto de documentos legales:
{context}

Pregunta: {question}

Respuesta detallada con referencias específicas:"""

    legal_prompt = PromptTemplate(
        template=legal_prompt_template,
        input_variables=["context", "question"]
    )
    
    def format_documents(docs):
        """Formatea documentos con información de metadata relevante"""
        formatted_docs = []
        for doc in docs:
            metadata = doc.metadata
            
            # Construir información de contexto
            source_info = f"Fuente: {metadata.get('source_file', 'Desconocido')}"
            page_info = f"Página: {metadata.get('page', 'N/A')}"
            
            context_parts = [source_info, page_info]
            
            if metadata.get('article_number'):
                context_parts.append(f"Artículo: {metadata['article_number']}")
            
            context_header = " | ".join(context_parts)
            formatted_doc = f"[{context_header}]\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n---\n".join(formatted_docs)
    
    # Construir cadena de procesamiento
    rag_chain = (
        {
            "context": lambda x: format_documents(retriever.get_relevant_documents(x["question"])),
            "question": RunnablePassthrough()
        }
        | legal_prompt
        | llm
        | StrOutputParser()
    )
    
    def run_rag_query(question: str) -> str:
        """Ejecuta una consulta RAG con manejo de errores"""
        try:
            result = rag_chain.invoke({"question": question})
            return result
        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"
    
    return Tool(
        name="rag_legal_documents",
        description="""Herramienta especializada para consultar documentos legales mexicanos (leyes, reglamentos, etc.). 
        Úsala para:
        - Encontrar artículos específicos por número
        - Buscar información sobre temas legales específicos
        - Obtener citas exactas de documentos legales
        - Consultar fracciones, incisos y párrafos específicos
        
        Ejemplos de consultas efectivas:
        - "¿Qué dice el artículo 15 sobre IMMEX?"
        - "Requisitos para la fracción III del artículo 25"
        - "Definición de manufactura en el reglamento"
        """,
        func=run_rag_query,
    )

# Función auxiliar para testing
def test_enhanced_rag(llm):
    """Función para probar el RAG mejorado"""
    test_queries = [
        "¿Qué dice el artículo 15 sobre IMMEX?",
        "Requisitos para empresas manufactureras",
        "¿En qué artículo se habla de importación temporal?",
        "Definición de insumos"
    ]
    
    tool = build_enhanced_rag_tool(llm)
    
    print("🧪 Probando RAG mejorado para documentos legales\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Pregunta: {query}")
        try:
            result = tool.func(query)
            print(f"   Respuesta: {result[:300]}...")
            print("-" * 50)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("-" * 50)

if __name__ == "__main__":
    # Ejemplo de uso/testing
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    test_enhanced_rag(llm)