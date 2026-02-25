import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import SkipGramModel

# ==========================================
# 1. LOAD SAVED DATA
# ==========================================
print("Loading vocabulary and model...")
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)

with open("idx2word.json", "r") as f:
    idx2word_str_keys = json.load(f)
    idx2word = {int(k): v for k, v in idx2word_str_keys.items()}

VOCAB_SIZE = len(word2idx)
EMBEDDING_DIM = 50 

model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
model.load_state_dict(torch.load("word2vec_model.pth"))
model.eval()

# ==========================================
# 2. SELECT WORDS TO PLOT
# ==========================================
# We pick words we expect to form distinct semantic clusters
words_to_visualize = [
    # Animals
    "cat", "dog", "fox", "wolf", "bear", "lion", "tiger", "fish", "bird",
    # Colors
    "red", "blue", "green", "yellow", "black", "white", "purple", "brown",
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    # Family
    "mother", "father", "sister", "brother", "son", "daughter", "uncle", "aunt",
    # Vehicles
    "car", "truck", "bus", "train", "ship", "boat", "plane", "bicycle"
]

# Filter out words that didn't make it into our training vocabulary
valid_words = [w for w in words_to_visualize if w in word2idx]
word_indices = [word2idx[w] for w in valid_words]

# Extract the 50-dimensional vectors for just these specific words
with torch.no_grad():
    tensor_indices = torch.tensor(word_indices)
    word_vectors_50d = model.center_embeddings(tensor_indices).numpy()

# ==========================================
# 3. REDUCE DIMENSIONS WITH t-SNE
# ==========================================
print("Running t-SNE to reduce from 50D to 2D...")
# perplexity controls the balance between local and global aspects of the data. 
# For small datasets, a lower perplexity (like 5-10) works best.
tsne = TSNE(n_components=2, random_state=42, perplexity=8, max_iter=1000)
word_vectors_2d = tsne.fit_transform(word_vectors_50d)

# ==========================================
# 4. PLOT THE GRAPH
# ==========================================
print("Generating plot...")
plt.figure(figsize=(12, 10))

# Plot each point and label it
for i, word in enumerate(valid_words):
    x, y = word_vectors_2d[i]
    plt.scatter(x, y, color='steelblue', s=50)
    plt.annotate(
        word, 
        (x, y), 
        xytext=(5, 2), 
        textcoords='offset points', 
        ha='right', 
        va='bottom',
        fontsize=12
    )

plt.title("Word2Vec Semantic Clusters (t-SNE Visualization)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot as an image file and display it
plt.savefig("word2vec_clusters.png", dpi=300, bbox_inches='tight')
print("Graph saved as 'word2vec_clusters.png'!")
plt.show()
