#pip install ollama scikit-learn
import os
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

product1_description = "OnePlus Buds Pro 3 Bluetooth TWS in-Earbuds Dual Drivers, Dual Dacs, Dynaudio Eqs, AI-Powered Translator, Up to 50Db Adaptive Noise Cancellation, Up to 43Hrs Battery."
product2_description = "OnePlus Bullets Z2 Bluetooth Wireless in Ear Earphones with Mic, Bombastic Bass - 12.4 Mm Drivers, 10 Mins Charge - 20 Hrs Music, 30 Hrs Battery Life "
product3_description = "BSB HOME® 100% Cotton Ultrasonic 280 Tc Solid King Size Quilted Bed Cover/Bedsheet with 2 Pillow Case, (Luxurious, Pink, 90X100 Inches, 254X228 cm)"

response = ollama.embed(
  model="mxbai-embed-large",
  input=product1_description
)
response1 = ollama.embed(
  model="mxbai-embed-large",
  input=product2_description
)
response2 = ollama.embed(
  model="mxbai-embed-large",
  input=product3_description
)
# Convert it into array ( sequence of numbers )
embed1 = np.array(response['embeddings']).reshape(1, -1)
embed2 = np.array(response1['embeddings']).reshape(1, -1)
embed3 = np.array(response2['embeddings']).reshape(1, -1)

print(cosine_similarity(embed1,embed2))
print(cosine_similarity(embed2,embed3))
print(cosine_similarity(embed3,embed1))
