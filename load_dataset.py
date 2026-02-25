import urllib.request
import zipfile
import os

def load_text8_dataset(max_words=1000000):
    """
    Downloads and loads the text8 Wikipedia dump.
    max_words: Limits the dataset size so it doesn't crash your computer 
               while testing. Set to None to load all 17 million words!
    """
    url = "http://mattmahoney.net/dc/text8.zip"
    file_name = "text8.zip"
    
    # 1. Download the file if we don't already have it
    if not os.path.exists("text8"):
        if not os.path.exists(file_name):
            print("Downloading text8 dataset (this might take a minute)...")
            urllib.request.urlretrieve(url, file_name)
            print("Download complete!")
            
        # 2. Extract the zip file
        print("Extracting...")
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall()
            
    # 3. Read the text file
    print("Loading text into memory...")
    with open("text8", "r") as f:
        text = f.read()
        
    # 4. Tokenize by simply splitting on spaces
    tokens = text.split()
    
    if max_words:
        tokens = tokens[:max_words]
        
    print(f"Loaded {len(tokens):,} words from Wikipedia!")
    return tokens

# --- How to use this in your vocab.py ---
# Instead of text_data = ["The quick brown fox..."]
# Do this:
# text8_tokens = load_text8_dataset(max_words=500000)
# word2idx, idx2word, all_tokens = build_vocab([" ".join(text8_tokens)], min_freq=5)
