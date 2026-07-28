"""
RAG pipeline module — Member 4 fills this in.

This file should contain the retrieval-augmented generation logic:
1. Load the FAISS index built from embedded recipe reference documents
2. Embed the incoming user query with the same embedding model
3. Retrieve the top-k most relevant recipes
4. Pass the retrieved context + query to an LLM
5. Return the generated, grounded response

Import this into app.py with:
    from rag_pipeline import rag_chat
and delete the placeholder rag_chat() function defined in app.py.
"""

# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import anthropic

# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# index = faiss.read_index("models/recipe_postings.index")
# client = anthropic.Anthropic()


def retrieve_relevant_recipes(query, top_k=5):
    """
    Embed the query and search the FAISS index for the most similar
    recipe reference documents. Return the matched text + metadata.
    """
    # query_embedding = embed_model.encode([query])
    # distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    # results = [...]
    # return results
    raise NotImplementedError("Wire this up to your FAISS index.")


def rag_chat(query, user_type="job_seeker"):
    """
    Full RAG pipeline: retrieve relevant recipes, then generate a grounded
    response using an LLM.
    """
    # retrieved = retrieve_relevant_recipes(query)
    # context = "\n\n---\n\n".join([r["chunk_text"] for r in retrieved])
    #
    # role_instruction = (
    #     "You are helping a home cook decide what to make."
    #     if user_type == "job_seeker"
    #     else "You are helping a recipe platform recommend content."
    # )
    #
    # prompt = f"""{role_instruction}
    #
    # Retrieved recipes:
    # {context}
    #
    # Question: {query}
    #
    # Answer using only the recipes above. If they don't answer it, say so."""
    #
    # response = client.messages.create(
    #     model="claude-sonnet-5",
    #     max_tokens=500,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text

    raise NotImplementedError("Wire this up to your LLM call.")
