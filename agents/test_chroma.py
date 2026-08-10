import chromadb


print("Connecting to ChromaDB...")


client = chromadb.HttpClient(
    host="localhost",
    port=8000
)
print("Connected")

collection = client.get_collection(
    name="recipes"
)


print("==============================")
print("Collection Name:")
print(collection.name)

print("==============================")
print("Total Documents:")
print(collection.count())


print("==============================")
print("Sample Recipe:")


data = collection.peek(
    limit=1
)


print(data)