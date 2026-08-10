import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():

    return psycopg2.connect(

        host="localhost",

        port=os.getenv("POSTGRES_PORT"),

        database=os.getenv("POSTGRES_DB"),

        user=os.getenv("POSTGRES_USER"),

        password=os.getenv("POSTGRES_PASSWORD")

    )


def search_recipes(keyword):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(

        """
        SELECT DISTINCT recipe_name
        FROM
        (
            SELECT recipe_name FROM recipes

            UNION

            SELECT recipe_name FROM user_reviews
        ) t

        WHERE recipe_name ILIKE %s

        ORDER BY recipe_name

        LIMIT 20
        """,

        (f"%{keyword}%",)

    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return [row[0] for row in rows]