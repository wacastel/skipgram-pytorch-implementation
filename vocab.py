import re
from collections import Counter
from load_dataset import load_text8_dataset # <-- NEW IMPORT

def build_vocab(corpus, min_freq=1):
    # 1. Tokenization: Split strings into clean, lowercase words
    tokens = []
    for sentence in corpus:
        # This simple regex grabs alphanumeric words and ignores punctuation
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokens.extend(words)
    
    # 2. Count how often each word appears
    word_counts = Counter(tokens)
    
    # 3. Initialize dictionaries
    # It is best practice to include an <UNK> (unknown) token for 
    # words the model encounters later that weren't in the training data.
    word2idx = {"<UNK>": 0}
    idx2word = {0: "<UNK>"}
    
    # 4. Populate the dictionaries
    idx = 1
    for word, count in word_counts.items():
        # We can filter out extremely rare words if we want to save memory
        if count >= min_freq:  
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
            
    return word2idx, idx2word, tokens

# --- UPDATED DATA LOADING ---
print("Fetching text8 dataset...")
# Load 500,000 words from the text8 dataset to prevent memory overload during testing
text8_tokens = load_text8_dataset(max_words=500000)

print("Building vocabulary...")
# We join the tokens back into a single string inside a list because 
# build_vocab expects a list of sentences/strings. We also set 
# min_freq=5 to filter out rare words and keep the vocab size manageable.
word2idx, idx2word, all_tokens = build_vocab([" ".join(text8_tokens)], min_freq=5)

# Test the output
print(f"Total Unique Words (Vocab Size): {len(word2idx):,}")
if 'fox' in word2idx:
    print(f"Index for 'fox': {word2idx.get('fox')}")
print(f"Word at index 4: {idx2word.get(4)}")


def generate_training_pairs(tokens, word2idx, window_size=2):
    """
    Slides a context window across a list of tokens to generate 
    (center_word_idx, context_word_idx) pairs for Skip-Gram training.
    """
    training_pairs = []
    
    # Iterate through each word in our tokenized corpus
    for i, target_word in enumerate(tokens):
        # 1. Look up the index of the center word
        # If the word isn't in our vocab, we use the <UNK> token index (0)
        center_idx = word2idx.get(target_word, 0)
        
        # 2. Define the boundaries of our sliding window
        # max() and min() prevent us from grabbing indices outside the list bounds
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # 3. Grab all the context words within the window
        for j in range(start, end):
            # We don't want to pair the center word with itself!
            if i != j:
                context_word = tokens[j]
                context_idx = word2idx.get(context_word, 0)
                
                # Append the (center, context) pair to our training data
                training_pairs.append((center_idx, context_idx))
                
    return training_pairs

# Generate the pairs
print("Generating training pairs... (this might take a few seconds)")
window_size = 2
skip_gram_pairs = generate_training_pairs(all_tokens, word2idx, window_size)

print(f"Total training pairs generated: {len(skip_gram_pairs):,}")
print("First 5 Training Pairs (Center ID, Context ID):")
for pair in skip_gram_pairs[:5]:
    print(pair)
