import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=8001
)

collections = client.list_collections()

print("Collections:")
for c in collections:
    print(c.name)