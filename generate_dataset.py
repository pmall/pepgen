import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def generate_dataset():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # TODO: Add dataset generation logic here
        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    generate_dataset()
