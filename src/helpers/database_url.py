# Datos de conexión
USER = "postgres"
PASSWORD = "contrasena"  # <-- reemplaza con tu contraseña real
HOST = "localhost"
PORT = "5432"
DB_NAME = "tools_agent_app"

DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"