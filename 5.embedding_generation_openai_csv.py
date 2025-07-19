#pip install ollama
#pip install openai
##pip install langchain-community
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
data = pd.read_csv("product_description.csv")
product_description = data["product_description"].tolist()
product_names = data["product_name"].tolist()
embedding_array = []
for product in product_description:
  response = ollama.embed(
    model="mxbai-embed-large",
    input=product
  )['embeddings']
  embedding_array.append(response)

embedding_matrix = np.vstack(embedding_array)

similarity_matrix = cosine_similarity(embedding_matrix)

similarity_df = pd.DataFrame(similarity_matrix, index=data["product_name"], columns=data["product_name"])

# Save or display
print(similarity_df.head())  # Display top-left corner
similarity_df.to_csv("product_similarity_matrix.csv")

top_n_similar = []
N = 5
for i, product in enumerate(product_names):
  # Get similarity scores for the current product
  sim_scores = similarity_matrix[i]

  # Get indices of top N+1 (because the product itself will be the most similar)
  top_indices = np.argsort(sim_scores)[::-1][1:N + 1]

  for rank, idx in enumerate(top_indices, start=1):
    top_n_similar.append({
      "product_name": product,
      "similar_product": product_names[idx],
      "similarity_score": round(sim_scores[idx], 4),
      "rank": rank
    })

# Convert to DataFrame and save
top_n_df = pd.DataFrame(top_n_similar)
top_n_df.to_csv("top_n_similar_products.csv", index=False)
