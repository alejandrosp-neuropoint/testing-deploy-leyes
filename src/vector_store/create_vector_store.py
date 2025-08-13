from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from src.helpers.database_url import DATABASE_URL

import os
import sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
if "OPENAI_API_KEY" not in os.environ:
    print("No se encuentra la API KEY del servicio LLM")
    sys.exit(1)

embeddings = OpenAIEmbeddings()

# Usar el nuevo nombre de colección para evitar conflictos
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="document_embeddings_v2",  # Nuevo nombre
    connection=DATABASE_URL,
    use_jsonb=True,
)

# Función para obtener o crear vector store con limpieza si es necesario
def get_or_create_vector_store(clean=False):
    """
    Obtiene el vector store existente o crea uno nuevo
    
    Args:
        clean (bool): Si True, limpia la colección existente antes de crear
    """
    global embeddings
    
    if clean:
        return PGVector(
            embeddings=embeddings,
            collection_name="document_embeddings_v2",
            connection=DATABASE_URL,
            use_jsonb=True,
            pre_delete_collection=True,  # Limpia la colección
        )
    else:
        return vector_store