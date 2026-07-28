# models/

Place your trained model files here:

- `best_model.pkl` — your best-performing classifier (from Member 1's comparison in Section 5)
- `tfidf_vectorizer.pkl` — the fitted TfidfVectorizer used during training
- `scaler.pkl` — the fitted StandardScaler, if used
- `recipe_postings.index` — the FAISS index built by Member 4 for the RAG pipeline

None of these files are committed to GitHub (see .gitignore) since they exceed
GitHub's size limits or are regenerated locally. Save them here after running
your training notebook, or download them from the team's shared Google Drive link.
