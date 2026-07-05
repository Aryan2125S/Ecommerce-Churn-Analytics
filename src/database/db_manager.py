"""
Database Manager Module.

This module is responsible for managing the connection to the underlying
MySQL (or SQLite) database using SQLAlchemy 2.0. It defines the explicit ORM
schema for the `customers` table, handles the automated creation of tables,
and provides robust, transactional methods to bulk insert (ingest) DataFrames 
and query them back for machine learning pipelines.

Usage:
    from src.database.db_manager import DatabaseManager
    
    db = DatabaseManager()
    db.initialize_schema()
    db.ingest_dataframe(df)
    df_loaded = db.fetch_all_as_dataframe()

Dependencies:
    - sqlalchemy (ORM and Engine)
    - pandas (for DataFrame mapping and fast to_sql/read_sql operations)
    - src.config.settings (for database URI)
    - src.utils.exceptions (for standardized error mapping)
    - src.utils.logger (for execution logging)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Date,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config.settings import settings
from src.utils.exceptions import DatabaseConnectionError, DatabaseQueryError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# SQLAlchemy ORM Schema Definition
# ===========================================================================


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class CustomerRecord(Base):
    """Explicit ORM model representing the 32-column TDD schema."""
    
    __tablename__ = "customers"

    # Primary Identifier
    customer_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Demographics & Geography
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Membership & Lifecycle Dates
    membership_type: Mapped[str] = mapped_column(String(50), nullable=False)
    registration_date: Mapped[str] = mapped_column(String(20), nullable=False)
    last_purchase_date: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Transactional & Monetary
    purchase_frequency: Mapped[float] = mapped_column(Float, nullable=False)
    total_orders: Mapped[int] = mapped_column(Integer, nullable=False)
    average_order_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_spend: Mapped[float] = mapped_column(Float, nullable=False)
    discount_usage: Mapped[int] = mapped_column(Integer, nullable=False)
    coupon_usage: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Digital Behavior
    session_duration: Mapped[float] = mapped_column(Float, nullable=False)
    browsing_time: Mapped[int] = mapped_column(Integer, nullable=False)
    pages_viewed: Mapped[int] = mapped_column(Integer, nullable=False)
    wishlist_items: Mapped[int] = mapped_column(Integer, nullable=False)
    cart_abandonment: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Logistics & Service
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)
    returned_orders: Mapped[int] = mapped_column(Integer, nullable=False)
    refund_amount: Mapped[float] = mapped_column(Float, nullable=False)
    support_calls: Mapped[int] = mapped_column(Integer, nullable=False)
    complaints: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Ratings & Value
    customer_rating: Mapped[float] = mapped_column(Float, nullable=False)
    delivery_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    customer_satisfaction: Mapped[int] = mapped_column(Integer, nullable=False)
    customer_lifetime_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Business Metrics & Target
    customer_segment: Mapped[str] = mapped_column(String(50), nullable=False)
    loyalty_score: Mapped[int] = mapped_column(Integer, nullable=False)
    churn: Mapped[int] = mapped_column(Integer, nullable=False)


# ===========================================================================
# Database Manager
# ===========================================================================


class DatabaseManager:
    """Manages the connection, schema creation, and DataFrame IO for the DB."""

    def __init__(self, uri: str | None = None) -> None:
        """Initialize the DB Manager.

        Args:
            uri: Database connection URI. Defaults to settings.database.uri.
        """
        self.uri = uri or settings.database.uri
        
        # Configure Engine pooling (echo=False to prevent massive console spam)
        try:
            self.engine: Engine = create_engine(
                self.uri,
                pool_pre_ping=True,
                echo=False,
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database engine configured successfully.")
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise DatabaseConnectionError(f"Engine creation failed: {e}") from e

    def test_connection(self) -> bool:
        """Ping the database to ensure it is reachable.

        Returns:
            True if connection is successful.

        Raises:
            DatabaseConnectionError: If connection cannot be established.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test passed.")
            return True
        except OperationalError as e:
            logger.error(f"Database operational error: {e}")
            raise DatabaseConnectionError(f"Database unreachable: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected database connection error: {e}")
            raise DatabaseConnectionError(f"Connection failed: {e}") from e

    def initialize_schema(self) -> None:
        """Create tables exactly matching the ORM models if they don't exist.

        Raises:
            DatabaseQueryError: If DDL execution fails.
        """
        logger.info("Initializing database schema...")
        self.test_connection()
        
        try:
            # Create all tables defined in Base metadata
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema initialized successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Schema initialization failed: {e}")
            raise DatabaseQueryError(f"Failed to create tables: {e}") from e

    def ingest_dataframe(self, df: pd.DataFrame, table_name: str = "customers") -> None:
        """Bulk insert a pandas DataFrame into the specified database table.

        Args:
            df: The pandas DataFrame to ingest.
            table_name: The target SQL table name.

        Raises:
            DatabaseQueryError: If the ingestion query fails (e.g., schema mismatch).
        """
        logger.info(f"Ingesting {len(df)} records into '{table_name}' table...")
        try:
            # Using chunksize=5000 for memory safety on large datasets
            # Using if_exists='replace' to idempotently refresh the table during generation
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists="replace",
                index=False,
                chunksize=5000,
            )
            logger.info(f"Successfully ingested {len(df)} records into '{table_name}'.")
        except ValueError as e:
            logger.error(f"DataFrame to SQL mapping error: {e}")
            raise DatabaseQueryError(f"Data insertion mapping failed: {e}") from e
        except Exception as e:
            logger.error(f"SQL bulk insert failed: {e}", exc_info=True)
            raise DatabaseQueryError(f"Bulk insert failed: {e}") from e

    def fetch_all_as_dataframe(self, table_name: str = "customers") -> pd.DataFrame:
        """Retrieve the full table content into a pandas DataFrame.

        Args:
            table_name: The table to query.

        Returns:
            A pandas DataFrame populated with all records.

        Raises:
            DatabaseQueryError: If the SELECT query fails.
        """
        logger.info(f"Fetching all records from '{table_name}'...")
        try:
            df = pd.read_sql_table(table_name=table_name, con=self.engine)
            logger.info(f"Successfully fetched {len(df)} records from '{table_name}'.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch dataframe from table '{table_name}': {e}")
            raise DatabaseQueryError(f"Table read failed: {e}") from e
