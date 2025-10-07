from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pathlib import Path

def db_connect():
    """
    Establishes and returns a SQLAlchemy Engine for the PostgreSQL database.
    Environment variables for DB credentials are loaded from `.env` or a global `.env` file.

    Returns:
        sqlalchemy.engine.Engine: A SQLAlchemy Engine object to interact with the database.
    """
    # Load local .env file if it exists
    load_dotenv()

    # Load global .env as a fallback
    global_env_path = Path("~/.config/envs/futi.env").expanduser()
    if global_env_path.exists():
        load_dotenv(dotenv_path=global_env_path)

    # Access environment variables
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # Ensure all required variables are set
    if not all([DB_HOST, DB_USER, DB_PASS, DB_NAME]):
        missing = [var for var, value in [("DB_HOST", DB_HOST), ("DB_USER", DB_USER), ("DB_PASS", DB_PASS), ("DB_NAME", DB_NAME)] if not value]
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    # Construct the database URL for SQLAlchemy
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Create and return the SQLAlchemy Engine
    engine = create_engine(db_url, pool_pre_ping=True)  # `pool_pre_ping` ensures connections are alive
    return engine
