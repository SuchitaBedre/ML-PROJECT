from chromadb import HttpClient


client = HttpClient(
    host="localhost",
    port=8000
)


try:

    client.delete_collection(
        name="recipes"
    )

    print("Old collection deleted")

except Exception as e:

    print(e)
    