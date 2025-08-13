import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging
import re

# Imports usando las librerías que ya tienes instaladas
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

# Para PDFs usaremos PyPDF2 directamente (más compatible)
import PyPDF2
from io import BytesIO

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentTextSplitter:
    """
    Text splitter especializado para documentos legales que preserva estructura
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patrones para detectar estructura legal
        self.article_pattern = re.compile(r'Art[íi]culo\s+(\d+)[°º]?\.?', re.IGNORECASE)
        self.section_pattern = re.compile(r'Secci[óo]n\s+([IVXLC]+|[Uu]na|[Dd]os|[Tt]res|\d+)', re.IGNORECASE)
        self.chapter_pattern = re.compile(r'Cap[íi]tulo\s+([IVXLC]+|[Uu]no|[Dd]os|[Tt]res|\d+)', re.IGNORECASE)
        self.fraction_pattern = re.compile(r'[IVXLCivxlc]+\)\s+|[abcdefg]\)\s+|\d+\)\s+', re.IGNORECASE)
    
    def extract_legal_metadata(self, text: str) -> Dict[str, Any]:
        """Extrae metadata específica de documentos legales"""
        metadata = {}
        
        # Buscar artículos
        articles = self.article_pattern.findall(text)
        if articles:
            metadata['articles'] = [int(art) for art in articles if art.isdigit()]
            metadata['article_mentions'] = len(articles)
        
        # Buscar secciones
        sections = self.section_pattern.findall(text)
        if sections:
            metadata['sections'] = sections
        
        # Buscar capítulos
        chapters = self.chapter_pattern.findall(text)
        if chapters:
            metadata['chapters'] = chapters
        
        # Contar fracciones
        fractions = self.fraction_pattern.findall(text)
        if fractions:
            metadata['fraction_count'] = len(fractions)
        
        return metadata
    
    def split_text_preserving_structure(self, text: str) -> List[Dict[str, Any]]:
        """Divide el texto preservando la estructura legal"""
        
        # Primero, intentar dividir por artículos
        article_splits = self.article_pattern.split(text)
        
        chunks = []
        current_article = None
        
        for i, split in enumerate(article_splits):
            if not split.strip():
                continue
            
            # Si es un número de artículo
            if split.isdigit() and i < len(article_splits) - 1:
                current_article = int(split)
                article_content = article_splits[i + 1] if i + 1 < len(article_splits) else ""
                
                # Procesar el contenido del artículo
                if article_content.strip():
                    # Si el artículo es muy largo, dividirlo
                    if len(article_content) > self.chunk_size:
                        sub_chunks = self.split_long_text(article_content)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                'text': f"Artículo {current_article}. {sub_chunk}",
                                'article_number': current_article,
                                'sub_chunk': j,
                                'is_article_start': j == 0
                            })
                    else:
                        chunks.append({
                            'text': f"Artículo {current_article}. {article_content}",
                            'article_number': current_article,
                            'sub_chunk': 0,
                            'is_article_start': True
                        })
            
            # Si no se detectaron artículos, usar división normal
            elif i == 0 and len(article_splits) == 1:
                normal_chunks = self.split_long_text(text)
                for j, chunk in enumerate(normal_chunks):
                    chunks.append({
                        'text': chunk,
                        'chunk_index': j,
                        'is_article_start': False
                    })
        
        return chunks
    
    def split_long_text(self, text: str) -> List[str]:
        """Divide texto largo en chunks con overlap inteligente"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # Buscar el mejor punto de corte
                for separator in ['\n\n', '. ', ';\n', ':\n', '\n', ' ']:
                    last_sep = text.rfind(separator, start, end)
                    if last_sep != -1:
                        end = last_sep + len(separator)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos preservando estructura legal"""
        split_docs = []
        
        for doc in documents:
            chunks_data = self.split_text_preserving_structure(doc.page_content)
            
            for chunk_data in chunks_data:
                # Extraer metadata legal del chunk
                legal_metadata = self.extract_legal_metadata(chunk_data['text'])
                
                new_doc = Document(
                    page_content=chunk_data['text'],
                    metadata={
                        **doc.metadata,
                        **chunk_data,
                        **legal_metadata,
                        "total_chunks": len(chunks_data)
                    }
                )
                split_docs.append(new_doc)
        
        return split_docs

def extract_text_from_pdf_enhanced(pdf_path: str) -> List[Document]:
    """
    Extrae texto de un PDF con mejor preservación de estructura
    """
    documents = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    
                    if text.strip():
                        # Limpieza más cuidadosa para documentos legales
                        clean_text_content = clean_legal_text(text)
                        
                        # Detectar si es página de índice o contenido
                        is_index_page = detect_index_page(clean_text_content)
                        
                        doc = Document(
                            page_content=clean_text_content,
                            metadata={
                                "source": pdf_path,
                                "source_file": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "total_pages": len(pdf_reader.pages),
                                "file_type": "pdf",
                                "is_index_page": is_index_page,
                                "character_count": len(clean_text_content)
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Error extrayendo texto de la página {page_num + 1} en {pdf_path}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error abriendo PDF {pdf_path}: {e}")
        raise
    
    return documents

def clean_legal_text(text: str) -> str:
    """
    Limpia texto de documentos legales preservando estructura importante
    """
    # Preservar saltos de línea importantes (antes de artículos, etc.)
    text = re.sub(r'\n\s*Art[íi]culo', '\n\nArtículo', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*Cap[íi]tulo', '\n\nCapítulo', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*Secci[óo]n', '\n\nSección', text, flags=re.IGNORECASE)
    
    # Normalizar espacios pero preservar párrafos
    text = re.sub(r'[ \t]+', ' ', text)  # Múltiples espacios -> uno solo
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Múltiples saltos -> doble salto
    
    # Preservar numeración importante
    text = re.sub(r'([IVXLC]+)\s*\)\s*', r'\1) ', text)  # Numeración romana
    text = re.sub(r'([abcdefg])\s*\)\s*', r'\1) ', text)  # Numeración alfabética
    
    # Limpiar caracteres problemáticos pero preservar acentos y signos de puntuación
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)ñáéíóúüÁÉÍÓÚÜÑ°º\[\]]', ' ', text)
    
    return text.strip()

def detect_index_page(text: str) -> bool:
    """Detecta si una página es un índice o tabla de contenidos"""
    index_indicators = [
        'índice', 'contenido', 'tabla de contenido',
        'página', 'pág', 'capítulo', 'sección'
    ]
    
    text_lower = text.lower()
    indicator_count = sum(1 for indicator in index_indicators if indicator in text_lower)
    
    # Si tiene muchos indicadores y es texto corto, probablemente es índice
    return indicator_count >= 2 and len(text) < 1000

def create_enhanced_vector_store(database_url: str, collection_name: str = "legal_documents_v3"):
    """
    Crea un vector store optimizado para documentos legales
    """
    from dotenv import load_dotenv, find_dotenv
    
    _ = load_dotenv(find_dotenv())
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("No se encuentra la API KEY del servicio LLM")
    
    # Usar modelo de embeddings más potente si está disponible
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Más preciso para documentos largos
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    
    return vector_store

def load_legal_pdfs_to_vector_store(
    data_directory: str = "data/",
    database_url: str = "None",
    collection_name: str = "legal_documents_v3",
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> PGVector:
    """
    Carga PDFs legales con procesamiento especializado
    """
    
    if not os.path.exists(data_directory):
        raise ValueError(f"El directorio {data_directory} no existe")
    
    # Crear vector store mejorado
    vector_store = create_enhanced_vector_store(database_url, collection_name)
    
    pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No se encontraron archivos PDF en {data_directory}")
        return vector_store
    
    logger.info(f"Encontrados {len(pdf_files)} archivos PDF")
    
    # Usar el splitter especializado para documentos legales
    text_splitter = LegalDocumentTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_documents = []
    
    for pdf_path in pdf_files:
        try:
            logger.info(f"Procesando: {pdf_path}")
            
            # Extraer con mejor preservación de estructura
            documents = extract_text_from_pdf_enhanced(pdf_path)
            
            if not documents:
                logger.warning(f"No se pudo extraer texto de {pdf_path}")
                continue
            
            # Filtrar páginas de índice si es necesario
            content_documents = [doc for doc in documents if not doc.metadata.get('is_index_page', False)]
            logger.info(f"Páginas de contenido: {len(content_documents)}/{len(documents)}")
            
            # Dividir con preservación de estructura legal
            split_documents = text_splitter.split_documents(content_documents)
            
            logger.info(f"Generados {len(split_documents)} chunks desde {os.path.basename(pdf_path)}")
            all_documents.extend(split_documents)
            
        except Exception as e:
            logger.error(f"Error procesando {pdf_path}: {str(e)}")
            continue
    
    # Cargar con mejor gestión de memoria
    if all_documents:
        try:
            logger.info(f"Cargando {len(all_documents)} documentos al vector store...")
            
            # Lotes más pequeños para documentos complejos
            batch_size = 30
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                vector_store.add_documents(batch)
                logger.info(f"Cargado lote {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}")
            
            logger.info("✅ Carga completada exitosamente")
        except Exception as e:
            logger.error(f"Error cargando documentos al vector store: {str(e)}")
            raise
    else:
        logger.warning("No se procesaron documentos")
    
    return vector_store

# Script principal
if __name__ == "__main__":
    from src.helpers.database_url import DATABASE_URL
    
    try:
        # Cargar con procesamiento especializado para documentos legales
        vector_store = load_legal_pdfs_to_vector_store(
            data_directory="data/",
            database_url=DATABASE_URL,
            collection_name="legal_documents_v3",
            chunk_size=800,  # Chunks más pequeños para mejor precisión
            chunk_overlap=150
        )
        
        # Pruebas mejoradas
        test_queries = [
            "artículo 15",
            "immex",
            "fracción III"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Probando búsqueda: '{query}'")
            results = vector_store.similarity_search(query, k=3)
            
            for i, doc in enumerate(results):
                metadata = doc.metadata
                article_info = f" - Artículo {metadata.get('article_number', 'N/A')}" if metadata.get('article_number') else ""
                logger.info(f"  {i+1}. {metadata.get('source_file', 'Unknown')} - Página {metadata.get('page', 'Unknown')}{article_info}")
                logger.info(f"     Preview: {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error en la carga: {str(e)}")
        raise