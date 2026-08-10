import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=8001
)

collection = client.get_collection("recipes")

print("Total Records:", collection.count())