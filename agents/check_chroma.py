import chromadb

client = chromadb.HttpClient(
    host="127.0.0.1",
    port=8000
)

collections = client.list_collections()

print("Collections:")
for c in collections:
    print(c.name)