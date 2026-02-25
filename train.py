import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from tqdm import tqdm  # <-- NEW IMPORT

# --- IMPORT FROM YOUR OTHER FILES ---
from model import SkipGramModel
# IMPORTANT: Make sure we import idx2word this time so we can save it!
from vocab import word2idx, idx2word, skip_gram_pairs 

# 1. Set Hyperparameters
EMBEDDING_DIM = 50     # Increased slightly for the larger text8 dataset
LEARNING_RATE = 0.05
EPOCHS = 3             # Dropped to 3 because text8 has millions of pairs!
VOCAB_SIZE = len(word2idx)

# 2. Instantiate the Model, Loss Function, and Optimizer
model = SkipGramModel(VOCAB_SIZE, EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# 3. Execute the Training Loop
print(f"\nStarting training on {len(skip_gram_pairs):,} pairs for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Wrap our dataset in tqdm for a progress bar
    # mininterval=0.5 prevents the bar from updating too fast and slowing down the loop
    progress_bar = tqdm(skip_gram_pairs, desc=f"Epoch {epoch + 1}/{EPOCHS}", mininterval=0.5)
    
    for center_idx, true_context_idx in progress_bar:
        
        # Step A: Clear gradients
        optimizer.zero_grad()
        
        # Step B: Train on POSITIVE Sample
        center_tensor = torch.tensor([center_idx])
        context_tensor = torch.tensor([true_context_idx])
        
        pos_score = model(center_tensor, context_tensor)
        pos_label = torch.tensor([1.0]) 
        pos_loss = criterion(pos_score, pos_label)
        
        # Step C: Train on NEGATIVE Sample
        random_idx = random.randint(1, VOCAB_SIZE - 1) 
        while random_idx == true_context_idx or random_idx == center_idx:
            random_idx = random.randint(1, VOCAB_SIZE - 1)
            
        neg_context_tensor = torch.tensor([random_idx])
        
        neg_score = model(center_tensor, neg_context_tensor)
        neg_label = torch.tensor([0.0])
        neg_loss = criterion(neg_score, neg_label)
        
        # Step D: Backpropagate
        loss = pos_loss + neg_loss
        loss.backward()
        
        # Step E: Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

print("\nTraining complete!")

# 4. Save the Model and Vocabulary
print("Saving model and vocabulary...")

# Save the PyTorch model weights (.pth is the standard PyTorch extension)
torch.save(model.state_dict(), "word2vec_model.pth")

# Save the dictionaries as JSON files so we can easily load them later
with open("word2idx.json", "w") as f:
    json.dump(word2idx, f)

with open("idx2word.json", "w") as f:
    json.dump(idx2word, f)
    
print("Successfully saved 'word2vec_model.pth', 'word2idx.json', and 'idx2word.json'!")
