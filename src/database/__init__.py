"""
Database Abstraction Package.

Provides a repository-pattern database manager that abstracts SQL engine
connectivity, schema management, data seeding, and DataFrame-based query
execution. Supports seamless toggling between SQLite (zero-config local
development) and MySQL (production deployment) backends via environment
configuration.

Modules:
    db_manager: Implements the DatabaseManager class with methods for
                engine initialization, connection pooling, schema creation,
                bulk data insertion from DataFrames, parameterized query
                execution, and result serialization to pandas DataFrames.

Dependencies:
    - sqlalchemy (for engine abstraction and connection management)
    - pymysql (for MySQL backend connectivity)
    - pandas (for DataFrame I/O with SQL)
    - src.config (for database credentials and engine toggle)
    - src.utils (for logging and exception handling)
"""
