import chromadb
from sentence_transformers import SentenceTransformer


print("Connecting to ChromaDB...")


client = chromadb.HttpClient(
    host="127.0.0.1",
    port=8001
)


collection = client.get_collection(
    name="recipes"
)


print("Chroma Collection Loaded")
print("Documents:", collection.count())



model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



def search_recipes(query, top_k=5):


    print("\nSearching ChromaDB")
    print("Query:", query)


    embedding = model.encode(
        query
    )


    results = collection.query(

        query_embeddings=[
            embedding.tolist()
        ],

        n_results=top_k

    )


    documents = results["documents"][0]


    print(
        "Recipes Found:",
        len(documents)
    )


    return documents