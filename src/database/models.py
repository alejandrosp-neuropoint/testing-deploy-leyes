from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

# Tabla intermedia: muchos a muchos entre Orden y Producto
orden_producto = Table(
    'orden_producto',
    Base.metadata,
    Column('orden_id', ForeignKey('orden.id'), primary_key=True),
    Column('producto_id', ForeignKey('producto.id'), primary_key=True),
    Column('cantidad', Integer, nullable=False)
)

class Producto(Base):
    __tablename__ = 'producto'

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)
    descripcion = Column(Text)
    precio = Column(Float, nullable=False)
    en_stock = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<Producto(nombre={self.nombre}, precio={self.precio})>"


class Cliente(Base):
    __tablename__ = 'cliente'

    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)

    ordenes = relationship("Orden", back_populates="cliente")

    def __repr__(self):
        return f"<Cliente(nombre={self.nombre}, email={self.email})>"


class Orden(Base):
    __tablename__ = 'orden'

    id = Column(Integer, primary_key=True)
    cliente_id = Column(Integer, ForeignKey('cliente.id'))
    fecha = Column(DateTime, default=datetime.utcnow)
    total = Column(Float)

    cliente = relationship("Cliente", back_populates="ordenes")
    productos = relationship("Producto", secondary=orden_producto)

    def __repr__(self):
        return f"<Orden(id={self.id}, fecha={self.fecha}, total={self.total})>"
