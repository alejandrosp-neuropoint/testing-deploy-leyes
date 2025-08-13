from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseLanguageModel
from src.helpers.database_url import DATABASE_URL



def build_sql_tool(llm: BaseLanguageModel):
    # Crea el engine SQLAlchemy y el wrapper para LangChain
    db = SQLDatabase.from_uri(DATABASE_URL)

    # Define la herramienta como un objeto Tool
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()  # Devuelve la herramienta SQL básica
