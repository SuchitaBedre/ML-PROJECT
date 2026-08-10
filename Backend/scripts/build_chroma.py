from pyspark.sql import SparkSession
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =================================
# Spark
# =================================

spark = SparkSession.builder \
    .appName("Build Chroma Resume") \
    .getOrCreate()


# =================================
# Load Dataset
# =================================

print("\nLoading Recipe Dataset...")

df = spark.read.parquet(
    "data/processed/final_recipe_dataset"
)


# =================================
# Chroma Connection
# =================================

client = HttpClient(
    host="localhost",
    port=8001
)


collection = client.get_or_create_collection(
    name="recipes",
    metadata={
        "hnsw:space": "cosine"
    }
)


# =================================
# Load Embedding Model
# =================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =================================
# Convert Spark Data
# =================================

rows = df.select(
    "recipe_id",
    "name",
    "ingredients",
    "steps",
    "description",
    "average_rating"
).collect()


total = len(rows)


print("\n=================================")
print("Total Recipes :", total)
print("=================================")



# =================================
# Resume Check
# =================================

existing_count = collection.count()


print(
    "Existing Chroma Records :",
    existing_count
)


if existing_count >= total:

    print(
        "\nAll recipes already stored!"
    )

    exit()


start_index = existing_count


print(
    f"\nResuming from recipe {start_index + 1}"
)



# =================================
# Batch Settings
# =================================

embedding_batch_size = 50000

# Chroma maximum allowed is 5461
chroma_batch_size = 5000



# =================================
# Build Chroma
# =================================

for start in tqdm(
        range(
            start_index,
            total,
            embedding_batch_size
        ),
        desc="Building ChromaDB",
        unit="batch"
):


    end = min(
        start + embedding_batch_size,
        total
    )


    batch_rows = rows[start:end]


    documents = []
    ids = []
    metadatas = []



    # ------------------------------
    # Prepare Documents
    # ------------------------------

    for row in batch_rows:


        document = f"""
Recipe Name:
{row['name']}


Description:
{row['description']}


Ingredients:
{row['ingredients']}


Steps:
{row['steps']}


Average Rating:
{row['average_rating']}
"""


        documents.append(document)


        ids.append(
            str(row["recipe_id"])
        )


        metadatas.append(
            {
                "recipe_name": row["name"],
                "rating": float(
                    row["average_rating"]
                )
            }
        )



    print(
        f"\nEncoding recipes {start+1} to {end}"
    )



    # ------------------------------
    # Generate Embeddings
    # ------------------------------

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True
    ).tolist()



    print(
        f"Saving recipes {start+1} to {end}"
    )



    # ------------------------------
    # Insert into Chroma
    # ------------------------------

    for i in range(
        0,
        len(ids),
        chroma_batch_size
    ):


        collection.add(

            ids=
            ids[i:i+chroma_batch_size],


            documents=
            documents[i:i+chroma_batch_size],


            embeddings=
            embeddings[i:i+chroma_batch_size],


            metadatas=
            metadatas[i:i+chroma_batch_size]

        )


        stored = min(
            i + chroma_batch_size,
            len(ids)
        )


        print(
            f"Stored {stored}/{len(ids)}"
        )



    print(
        "\nCompleted:",
        end,
        "/",
        total,
        "recipes\n"
    )



print(
    "\n================================="
)

print(
    "ALL RECIPES INSERTED SUCCESSFULLY"
)

print(
    "Final Count:",
    collection.count()
)

print(
    "================================="
)



spark.stop()