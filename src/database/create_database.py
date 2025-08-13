from sqlalchemy import create_engine
from src.database.models import Base
from src.helpers.database_url import DATABASE_URL

# Crear el engine
engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    print("Creando tablas de la base de datos")
    Base.metadata.create_all(engine)
    print("Tablas creadas correctamente")