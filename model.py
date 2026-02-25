import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        
        # Matrix V: The representations when words are the CENTER word
        # Shape: (vocab_size, embedding_dim)
        self.center_embeddings = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim
        )
        
        # Matrix U: The representations when words are the CONTEXT word
        # Shape: (vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim
        )
        
        # Initialize the weights with small random values
        # This is a common practice to help the model start learning effectively
        initrange = 0.5 / embedding_dim
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_word_idx, context_word_idx):
        """
        The forward pass calculates the similarity between the center 
        word and the context word.
        """
        # 1. Look up the dense vector for the center word (v_c)
        # Shape: (batch_size, embedding_dim)
        v_c = self.center_embeddings(center_word_idx)
        
        # 2. Look up the dense vector for the context word (u_o)
        # Shape: (batch_size, embedding_dim)
        u_o = self.context_embeddings(context_word_idx)
        
        # 3. Calculate the dot product to get the similarity score
        # We multiply the vectors element-wise and sum them up
        # A higher score means the model thinks these words belong together
        score = torch.sum(v_c * u_o, dim=1)
        
        return score
