import torch
import torch.nn.functional as F
import json
from model import SkipGramModel

# ==========================================
# 1. LOAD SAVED DATA
# ==========================================
print("Loading vocabularies...")
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)

with open("idx2word.json", "r") as f:
    # JSON converts dict keys to strings. We must cast them back to integers!
    idx2word_str_keys = json.load(f)
    idx2word = {int(k): v for k, v in idx2word_str_keys.items()}

print("Loading the trained model...")
VOCAB_SIZE = len(word2idx)
EMBEDDING_DIM = 50  # This MUST match the dimension you set in train.py

# Instantiate the blank architecture, then load the saved weights into it
model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
model.load_state_dict(torch.load("word2vec_model.pth"))

# Lock the model for evaluation (disables gradient tracking overhead)
model.eval()


# ==========================================
# 2. EVALUATION FUNCTIONS
# ==========================================
def get_similarity(word1, word2, model, word2idx):
    """Calculates cosine similarity between two words (-1 to 1)."""
    if word1 not in word2idx or word2 not in word2idx:
        return f"Error: '{word1}' or '{word2}' is not in the vocabulary!"
        
    idx1 = torch.tensor([word2idx[word1]])
    idx2 = torch.tensor([word2idx[word2]])
    
    with torch.no_grad():
        vec1 = model.center_embeddings(idx1)
        vec2 = model.center_embeddings(idx2)
        
    similarity = F.cosine_similarity(vec1, vec2)
    return similarity.item()


def find_analogy(word_a, word_b, word_c, model, word2idx, idx2word):
    """Solves: A is to B as C is to ?"""
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            return f"Error: '{w}' is not in the vocabulary.", None

    idx_a = torch.tensor([word2idx[word_a]])
    idx_b = torch.tensor([word2idx[word_b]])
    idx_c = torch.tensor([word2idx[word_c]])

    with torch.no_grad():
        vec_a = model.center_embeddings(idx_a)
        vec_b = model.center_embeddings(idx_b)
        vec_c = model.center_embeddings(idx_c)
        
        # The core vector arithmetic: target = B - A + C
        target_vec = vec_b - vec_a + vec_c

        best_word = None
        best_sim = -2.0 

        for word, idx in word2idx.items():
            if word in [word_a, word_b, word_c, "<UNK>"]:
                continue
                
            vec_w = model.center_embeddings(torch.tensor([idx]))
            sim = F.cosine_similarity(target_vec, vec_w).item()
            
            if sim > best_sim:
                best_sim = sim
                best_word = word

    return best_word, best_sim


# ==========================================
# 3. TEST THE MODEL
# ==========================================
print("\n--- Evaluating Learned Embeddings ---")

# Test Similarity
word_pairs_to_test = [
    ("one", "two"),
    ("king", "queen"),
    ("apple", "computer")
]

print("\nCosine Similarities:")
for w1, w2 in word_pairs_to_test:
    score = get_similarity(w1, w2, model, word2idx)
    # If the words exist, print the score, otherwise print the error
    if isinstance(score, float):
        print(f"  '{w1}' & '{w2}': {score:.4f}")
    else:
        print(f"  {score}")

# Test Analogies
analogies_to_test = [
    ("man", "king", "woman"),
    ("paris", "france", "london"),
    ("walk", "walking", "run")
]

print("\nSolving Analogies:")
for a, b, c in analogies_to_test:
    best_word, score = find_analogy(a, b, c, model, word2idx, idx2word)
    print(f"  {a} : {b} :: {c} : {best_word} (score: {score:.4f} if valid)")
