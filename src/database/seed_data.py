from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Producto, Cliente, Orden, orden_producto
from datetime import datetime
from src.helpers.database_url import DATABASE_URL

engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)
session = Session()

def seed_data():
    # Evitar duplicación si ya se corrió antes
    if session.query(Producto).count() > 0:
        print("Los datos ya fueron insertados.")
        return

    # Productos
    p1 = Producto(nombre="Computadora Gamer", descripcion="Intel i7, RTX 4070, 32GB RAM", precio=25000, en_stock=10)
    p2 = Producto(nombre="Laptop de oficina", descripcion="AMD Ryzen 5, 16GB RAM", precio=14000, en_stock=15)
    p3 = Producto(nombre="Mini PC", descripcion="Intel NUC, 8GB RAM", precio=8000, en_stock=20)

    session.add_all([p1, p2, p3])

    # Clientes
    c1 = Cliente(nombre="Juan Pérez", email="juan.perez@example.com")
    c2 = Cliente(nombre="Laura Gómez", email="laura.gomez@example.com")

    session.add_all([c1, c2])

    # Órdenes
    o1 = Orden(cliente=c1, fecha=datetime(2025, 6, 1), total=p1.precio + p3.precio)
    o2 = Orden(cliente=c2, fecha=datetime(2025, 6, 2), total=p2.precio)

    session.add_all([o1, o2])
    
    session.flush()

    # Asociar productos a órdenes (tabla intermedia)
    session.execute(orden_producto.insert().values([
        {"orden_id": o1.id, "producto_id": p1.id, "cantidad": 1},
        {"orden_id": o1.id, "producto_id": p3.id, "cantidad": 2},
        {"orden_id": o2.id, "producto_id": p2.id, "cantidad": 1}
    ]))

    # Guardar cambios
    session.commit()
    print("Datos de ejemplo insertados correctamente.")

if __name__ == "__main__":
    seed_data()
