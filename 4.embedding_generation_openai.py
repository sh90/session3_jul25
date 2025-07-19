#pip install ollama scikit-learn
#pip install tiktoken
##pip install langchain-community
import os
import ollama

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings

import data_info

OPENAI_API_KEY = data_info.open_ai_key

product1_description = "OnePlus Buds Pro 3 Bluetooth TWS in-Earbuds Dual Drivers, Dual Dacs, Dynaudio Eqs, AI-Powered Translator, Up to 50Db Adaptive Noise Cancellation, Up to 43Hrs Battery."
product2_description = "OnePlus Bullets Z2 Bluetooth Wireless in Ear Earphones with Mic, Bombastic Bass - 12.4 Mm Drivers, 10 Mins Charge - 20 Hrs Music, 30 Hrs Battery Life "
product3_description = "BSB HOME® 100% Cotton Ultrasonic 280 Tc Solid King Size Quilted Bed Cover/Bedsheet with 2 Pillow Case, (Luxurious, Pink, 90X100 Inches, 254X228 cm)"

EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)

response= embedding_model.embed_query(product1_description)
response1 = embedding_model.embed_query(product2_description)
response2 = embedding_model.embed_query(product3_description)

# Convert it into array ( sequence of numbers )
embed1 = np.array(response).reshape(1, -1)
embed2 = np.array(response1).reshape(1, -1)
embed3 = np.array(response2).reshape(1, -1)

print(cosine_similarity(embed1,embed2))
print(cosine_similarity(embed2,embed3))
print(cosine_similarity(embed3,embed1))
