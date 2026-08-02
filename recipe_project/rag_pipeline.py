"""
RAG pipeline — retrieval-based recipe assistant using ChromaDB.

Uses 230,185 real recipes with nutrition metadata, embedded via
Sentence-Transformers. Includes a lightweight memory layer for
follow-up queries, and caps implausible nutrition values (raw
source data outliers) at display time.
"""

import chromadb
import os

CHROMA_DIR = os.path.join("models", "chroma_db_v3")

_client = None
_collection = None
_embed_model = None

FOLLOWUP_WORDS = ['it', 'that', 'this', 'each', 'these', 'those', 'them', 'they']
NUTRITION_KEYWORDS = ['calorie', 'calories', 'fat', 'protein', 'sugar', 'carb', 'carbs',
                      'carbohydrate', 'healthy', 'nutrition', 'sodium']

# Reasonable per-serving caps — raw dataset has some outliers
# (e.g. one recipe listed 223g protein, 1200+ calories, likely a
# batch recipe or data entry issue in the original source)
CAPS = {
    'calories': 1500,
    'protein': 100,
    'total_fat': 100,
    'carbohydrates': 150,
    'sugar': 80,
}


def get_embed_model():
    """Load the same embedding model used to build the index."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def get_collection():
    """Load the ChromaDB collection once, reuse across calls."""
    global _client, _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            return None
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_collection(name="recipes")
    return _collection


def is_followup(query, has_history):
    """Heuristic: short query containing a reference word, AND we have
    prior results to refer back to."""
    if not has_history:
        return False
    words = query.lower().strip().split()
    return len(words) <= 6 and any(w in FOLLOWUP_WORDS for w in words)


def wants_nutrition(query):
    return any(w in query.lower() for w in NUTRITION_KEYWORDS)


def agent_1_router(query, has_history=False):
    """Decide what kind of request this is."""
    query_lower = query.lower().strip()
    words = query_lower.split()

    greeting_keywords = ['hello', 'hi', 'hey', 'thanks', 'thank']

    if len(words) <= 3 and any(w in greeting_keywords for w in words):
        return "greeting"
    elif is_followup(query, has_history):
        return "followup"
    elif wants_nutrition(query):
        return "nutrition_search"
    else:
        return "recipe_search"


def agent_2_retrieve(query, collection, n_results=5, relevance_threshold=1.6):
    """Embed the query with the same model used at index time, then search."""
    embed_model = get_embed_model()
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    filtered = []
    for doc, meta, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        if distance < relevance_threshold:
            filtered.append({'doc': doc, 'meta': meta, 'distance': distance})

    return filtered


def capped(meta, field):
    """Cap a nutrition value at a reasonable maximum for display."""
    value = meta.get(field, 0) or 0
    cap = CAPS.get(field)
    return min(value, cap) if cap else value


def format_nutrition(retrieved_items):
    response = "Here's the nutrition info for those recipes:\n\n"
    for i, item in enumerate(retrieved_items[:3], 1):
        meta = item['meta']
        response += (
            f"**{i}. {meta.get('name', 'Unknown')}**\n"
            f"   Calories: {capped(meta, 'calories'):.0f} | "
            f"Protein: {capped(meta, 'protein'):.0f}g | "
            f"Fat: {capped(meta, 'total_fat'):.0f}g | "
            f"Carbs: {capped(meta, 'carbohydrates'):.0f}g | "
            f"Sugar: {capped(meta, 'sugar'):.0f}g\n\n"
        )
    return response


def format_recipe_list(retrieved_items, query):
    response = f"Based on your search for **\"{query}\"**, here are the best matches:\n\n"
    for i, item in enumerate(retrieved_items[:3], 1):
        meta = item['meta']
        ingredients = meta.get('ingredients', '')[:120]
        response += (
            f"**{i}. {meta.get('name', 'Unknown')}**\n"
            f"   Ingredients: {ingredients}...\n"
            f"   {capped(meta, 'calories'):.0f} cal, {capped(meta, 'protein'):.0f}g protein\n\n"
        )
    return response


def agent_3_format_response(query, route, retrieved_items):
    """Structure the final response based on what was found."""
    if route == "greeting":
        return "Hello! Ask me about recipes, ingredients, nutrition, or what to cook based on what you have on hand."

    if not retrieved_items:
        return (
            f"I couldn't find a strong match for \"{query}\" in our 230,185 recipes. "
            "Try describing the dish differently, or mention specific ingredients."
        )

    if route in ("nutrition_search", "followup") or wants_nutrition(query):
        return format_nutrition(retrieved_items)

    return format_recipe_list(retrieved_items, query)


def rag_chat(query, last_retrieved=None):
    """
    Full retrieval pipeline with lightweight memory.

    Args:
        query: the user's current message
        last_retrieved: list of recipes retrieved in the previous turn (or None)

    Returns:
        (response_text, new_last_retrieved) — pass new_last_retrieved back in
        on the next call to preserve context.
    """
    collection = get_collection()
    if collection is None:
        return (
            "The recipe database isn't loaded yet. Make sure the models/chroma_db_v3 "
            "folder is present.",
            last_retrieved
        )

    has_history = bool(last_retrieved)
    route = agent_1_router(query, has_history)

    if route == "greeting":
        return agent_3_format_response(query, route, []), last_retrieved

    if route == "followup":
        # Reuse previous results rather than searching fresh
        response = agent_3_format_response(query, route, last_retrieved)
        return response, last_retrieved

    retrieved = agent_2_retrieve(query, collection)
    response = agent_3_format_response(query, route, retrieved)
    return response, retrieved