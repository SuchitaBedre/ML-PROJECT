import psycopg2



def get_connection():

    return psycopg2.connect(

        host="localhost",

        database="recipe_db",

        user="postgres",

        password="recipe123",

        port=5432

    )