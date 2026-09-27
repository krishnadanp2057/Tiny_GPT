# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import torch
# Import PyTorch.
# PyTorch is used to create tensors, neural networks, train the model, etc.


from transformer_block import Block, MultiHeadAttention, FeedForward
# Import the Transformer components from your transformer_block.py file.
#
# Block             → one Transformer block
# MultiHeadAttention → attention mechanism
# FeedForward       → feed-forward neural network inside Transformer
#
# TinyGPT will use Block() later.


import torch.nn as nn
# Import PyTorch's neural-network module.
# nn contains things like:
# nn.Module
# nn.Embedding
# nn.Linear
# nn.LayerNorm
# etc.


import torch.nn.functional as F
# Import PyTorch's functional operations.
# We use:
# F.cross_entropy() → calculate prediction error/loss
# F.softmax()       → convert scores into probabilities


# ============================================================
# 2. CHECK PYTORCH / APPLE MPS
# ============================================================

print(torch.__version__)
# Print the installed PyTorch version.
#
# Example:
# 2.6.0
#
# This tells you which version of PyTorch you are using.


print(torch.backends.mps.is_available())
# Check whether Apple's MPS (Metal Performance Shaders) is available.
#
# MPS allows PyTorch to use the Apple GPU.
#
# True  → MPS GPU can be used
# False → MPS GPU is not available


print(torch.backends.mps.is_built())
# Check whether the installed PyTorch was built with MPS support.
#
# True → PyTorch was built with MPS support.
# False → PyTorch does not have MPS support.


# ============================================================
# 3. CREATE THE TRAINING DATA
# ============================================================

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]
# This is your training dataset.
#
# You have 10 sentences.
#
# The model will learn patterns from these sentences.
#
# Example:
# "hello friends how are you"
#
# The model tries to learn:
# hello → friends
# friends → how
# how → are
# are → you


corpus = [s + " <END>" for s in corpus]
# Add <END> to the end of every sentence.
#
# Before:
# "hello friends how are you"
#
# After:
# "hello friends how are you <END>"
#
# Why?
# <END> tells the model:
# "This sentence has finished."
#
# So the model can learn where one sentence ends.


text = " ".join(corpus)
# Join all 10 sentences into ONE large string.
#
# Example:
#
# Sentence 1 <END> Sentence 2 <END> Sentence 3 <END> ...
#
# So instead of having 10 separate strings,
# we now have one long piece of text.


print(text)
# Display the complete training text.


# ============================================================
# 4. CREATE VOCABULARY
# ============================================================

word = list(set(text.split()))
# text.split() separates the text into individual tokens/words.
#
# Example:
#
# "hello friends how are you"
#
# becomes:
#
# ["hello", "friends", "how", "are", "you"]
#
# set(...) removes duplicate words.
#
# For example:
#
# ["the", "tea", "the", "is"]
#
# becomes:
#
# {"the", "tea", "is"}
#
# list(...) converts the set back into a list.
#
# IMPORTANT:
# Because you are using set(), the order is not guaranteed.
# Therefore token IDs can be different each time you run the program.


vocab_size = len(word)
# Count how many UNIQUE tokens are in the vocabulary.
#
# If there are 42 unique tokens:
#
# vocab_size = 42
#
# IMPORTANT:
# Vocabulary size = UNIQUE tokens.
# It is NOT the total number of tokens in your dataset.


print(f"Vocabulary size: {vocab_size}")
# Print the number of unique tokens.
#
# Example:
# Vocabulary size: 42


word2idx = {w: i for i, w in enumerate(word)}
# Create a dictionary that converts WORD → NUMBER.
#
# Example:
#
# {
#     "hello": 33,
#     "friends": 27,
#     "how": 8,
#     ...
# }
#
# Why?
# Neural networks cannot directly work with words.
# They work with numbers.
#
# So:
#
# "hello" → 33
# "friends" → 27
# "how" → 8


print(word2idx)
# Display the word-to-number dictionary.


# ============================================================
# 5. CREATE REVERSE DICTIONARY
# ============================================================

idx2word = {i: w for w, i in word2idx.items()}
# Create the reverse dictionary.
#
# word2idx:
#
# "hello" → 33
#
# idx2word:
#
# 33 → "hello"
#
# We need this later to convert the model's generated
# numbers back into words.


# ============================================================
# 6. CONVERT ENTIRE TEXT INTO TOKEN IDs
# ============================================================

data = torch.tensor(
    [word2idx[w] for w in text.split()],
    dtype=torch.long
)
# Convert every word/token into its numerical ID.
#
# Example:
#
# text:
# hello friends how are you
#
# word2idx:
# hello   → 33
# friends → 27
# how     → 8
# are     → 1
# you     → 26
#
# data becomes:
#
# [33, 27, 8, 1, 26]
#
# torch.tensor() converts this list into a PyTorch tensor.
#
# dtype=torch.long is used because Embedding layers
# need integer token IDs.


print(data)
# Print the numerical representation of the entire text.


print(len(data))
# Print the TOTAL number of tokens in the dataset.
#
# IMPORTANT:
#
# vocab_size = unique tokens
#
# len(data) = total tokens, including repeated tokens
#
# They are NOT the same thing.


# ============================================================
# 7. MODEL HYPERPARAMETERS
# ============================================================

block_size = 6
# The model can look at a maximum of 6 tokens at a time.
#
# Example:
#
# [hello, friends, how, are, you, the]
#
# contains 6 tokens.
#
# block_size = 6 means sequence length = 6.


embedding_dim = 32
# Every token is represented using 32 numbers.
#
# Example:
#
# "hello"
#     ↓
# [0.21, -0.45, 0.73, ..., 0.15]
#
# There are 32 numbers.


n_heads = 2
# Each Transformer block has 2 attention heads.
#
# Attention heads allow the model to look at relationships
# between tokens in different ways.


n_layers = 2
# Use 2 Transformer blocks.
#
# x
# ↓
# Block 1
# ↓
# Block 2
# ↓
# output


lr = 1e-3
# Learning rate = 0.001.
#
# It controls how big the model's parameter updates are
# during learning.


epochs = 1500
# Your training loop will perform 1500 optimization steps.
#
# In THIS code, this is more accurately 1500 training steps,
# because each step uses one randomly selected batch.
#
# It is not necessarily 1500 complete passes over the dataset.


# ============================================================
# 8. CREATE TRAINING BATCHES
# ============================================================

def get_batch(batch_size=16):
    # Create a random batch of training examples.
    #
    # batch_size = 16 means:
    # "Give me 16 sequences at once."


    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Randomly select 16 starting positions from the dataset.
    #
    # Suppose:
    #
    # len(data) = 52
    # block_size = 6
    #
    # We have:
    #
    # 52 - 6 = 46
    #
    # So random starting positions are selected from
    # the available positions.
    #
    # Example:
    #
    # ix = [3, 15, 7, 20, ...]
    #
    # Each number says:
    # "Start a training sequence from this position."


    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create the INPUT sequences.
    #
    # If i = 3 and block_size = 6:
    #
    # data[3:9]
    #
    # gives 6 tokens.
    #
    # Example:
    #
    # x = [33, 27, 8, 1, 26, 24]
    #
    # This might represent:
    #
    # hello friends how are you the
    #
    # Because batch_size = 16:
    #
    # x.shape = [16, 6]


    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Create the TARGET sequences.
    #
    # y is shifted one token forward compared with x.
    #
    # Example:
    #
    # x:
    # hello friends how are you the
    #
    # y:
    # friends how are you the tea
    #
    # So the model learns:
    #
    # hello   → friends
    # friends → how
    # how     → are
    # are     → you
    # you     → the
    # the     → tea


    return x, y
    # Return:
    #
    # x = input
    # y = correct next-token answer


# ============================================================
# 9. CREATE THE TINY GPT MODEL
# ============================================================

class TinyGPT(nn.Module):
    # Create a neural network called TinyGPT.
    # nn.Module is PyTorch's base class for neural networks.


    def __init__(self):
        # This function runs when we create:
        #
        # model = TinyGPT()
        #
        # It creates all the layers of the model.


        super().__init__()
        # Initialize the parent nn.Module class.
        # This allows PyTorch to properly manage the model's
        # parameters and layers.


        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        # Convert token IDs into 32-dimensional vectors.
        #
        # If:
        # vocab_size = 42
        # embedding_dim = 32
        #
        # Then:
        #
        # 42 possible tokens
        # ↓
        # each token gets 32 numbers.
        #
        # Example:
        #
        # hello → 33 → [32 numbers]


        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )
        # Create embeddings for token positions.
        #
        # block_size = 6
        #
        # Positions:
        #
        # 0
        # 1
        # 2
        # 3
        # 4
        # 5
        #
        # Each position gets a 32-number vector.
        #
        # This tells the model WHERE each token occurs.


        self.blocks = nn.Sequential(
            *[
                Block(embedding_dim, block_size, n_heads)
                for _ in range(n_layers)
            ]
        )
        # Create multiple Transformer blocks.
        #
        # n_layers = 2
        #
        # Therefore:
        #
        # Block 1
        # ↓
        # Block 2
        #
        # The blocks learn relationships between tokens.
        #
        # n_heads = 2
        # means each block uses 2 attention heads.


        self.ln_f = nn.LayerNorm(embedding_dim)
        # Final Layer Normalization.
        #
        # It normalizes the representation before the final
        # prediction layer.


        self.head = nn.Linear(
            embedding_dim,
            vocab_size
        )
        # Final prediction layer.
        #
        # If:
        # embedding_dim = 32
        # vocab_size = 42
        #
        # Then:
        #
        # 32 numbers
        # ↓
        # 42 scores
        #
        # One score for each possible token in the vocabulary.


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, idx, targets=None):
        # Tell the model how to process the input.
        #
        # idx = input token IDs
        # targets = correct next token IDs
        #
        # Example:
        #
        # idx:
        # [hello, friends, how, are, you, the]
        #
        # targets:
        # [friends, how, are, you, the, tea]


        B, T = idx.shape
        # Get the input dimensions.
        #
        # B = batch size
        # T = number of tokens in each sequence
        #
        # Example:
        #
        # idx.shape = [16, 6]
        #
        # B = 16
        # T = 6


        tok_emb = self.token_embedding(idx)
        # Convert token IDs into vectors.
        #
        # Before:
        # [16, 6]
        #
        # After:
        # [16, 6, 32]
        #
        # 16 = sequences
        # 6  = tokens per sequence
        # 32 = numbers representing each token


        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )
        # Create position vectors.
        #
        # If T = 6:
        #
        # torch.arange(6)
        #
        # gives:
        #
        # [0, 1, 2, 3, 4, 5]
        #
        # These positions are converted into 32-number vectors.
        #
        # device=idx.device makes sure the position tensor
        # is on the same device as idx.


        x = tok_emb + pos_emb
        # Combine:
        #
        # token information
        # +
        # position information
        #
        # In simple words:
        #
        # "WHAT is the token?"
        # +
        # "WHERE is the token?"
        #
        # Result:
        # x.shape = [16, 6, 32]


        x = self.blocks(x)
        # Send x through the Transformer blocks.
        #
        # If n_layers = 2:
        #
        # x
        # ↓
        # Block 1
        # ↓
        # Block 2
        # ↓
        # x
        #
        # The Transformer learns relationships between tokens.


        x = self.ln_f(x)
        # Apply final Layer Normalization.
        #
        # Shape remains:
        # [16, 6, 32]


        logits = self.head(x)
        # Convert the 32-dimensional representation into
        # scores for every vocabulary token.
        #
        # If vocab_size = 42:
        #
        # [16, 6, 32]
        #       ↓
        # [16, 6, 42]
        #
        # 42 scores = 42 possible next tokens.


        loss = None
        # Initially there is no loss.


        if targets is not None:
            # If correct answers are available,
            # calculate how wrong the model's predictions are.


            B, T, C = logits.shape
            # Get dimensions of logits.
            #
            # Example:
            # logits.shape = [16, 6, 42]
            #
            # B = 16
            # T = 6
            # C = 42


            loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )
            # Calculate prediction error.
            #
            # B*T = 16 × 6 = 96
            #
            # logits:
            # [16, 6, 42]
            #
            # becomes:
            # [96, 42]
            #
            # This means:
            # 96 predictions
            # and each prediction has 42 possible choices.
            #
            # targets:
            # [16, 6]
            #
            # becomes:
            # [96]
            #
            # Cross entropy compares:
            #
            # model prediction
            #       VS
            # correct answer
            #
            # and produces a loss value.


        return logits, loss
        # Return:
        #
        # logits = model's predictions
        # loss   = model's error


    # ========================================================
    # TEXT GENERATION
    # ========================================================

    def generate(self, idx, max_new_tokens):
        # Generate new tokens one by one.
        #
        # idx = starting text
        # max_new_tokens = number of new tokens to generate.


        for _ in range(max_new_tokens):
            # Repeat the generation process.
            #
            # If max_new_tokens = 15:
            # generate 15 new tokens.
            #
            # "_" means:
            # We don't need the loop counter.


            idx_cond = idx[:, -block_size:]
            # Take only the LAST block_size tokens.
            #
            # If block_size = 6 and current sequence is:
            #
            # [A B C D E F G H]
            #
            # take:
            #
            # [C D E F G H]
            #
            # Why?
            # The model can only look at a maximum of 6 tokens.


            logits, _ = self(idx_cond)
            # Send the current context through TinyGPT.
            #
            # The model gives:
            #
            # logits → predictions
            # loss   → error
            #
            # We don't need the loss during generation,
            # so we store it in "_"


            logits = logits[:, -1, :]
            # Take only the prediction from the LAST position.
            #
            # Why?
            #
            # We want:
            #
            # "What word should come NEXT?"
            #
            # We don't need the predictions for earlier positions.


            probs = F.softmax(logits, dim=-1)
            # Convert raw logits/scores into probabilities.
            #
            # Example:
            #
            # tea      → 0.60
            # samosas  → 0.20
            # cricket  → 0.10
            # Delhi    → 0.10
            #
            # Total = 1.0


            next_idx = torch.multinomial(probs, 1)
            # Choose ONE next token using the probabilities.
            #
            # A token with a higher probability has a higher chance
            # of being selected.


            idx = torch.cat((idx, next_idx), dim=1)
            # Add the newly selected token to the end of the
            # existing sequence.
            #
            # Before:
            # [hello, friends]
            #
            # New token:
            # [how]
            #
            # After:
            # [hello, friends, how]


        return idx
        # Return the complete sequence after generating
        # all requested new tokens.


# ============================================================
# 10. CREATE THE MODEL
# ============================================================

model = TinyGPT()
# Create an actual TinyGPT model.
#
# This creates:
#
# Token Embedding
# Position Embedding
# Transformer Blocks
# LayerNorm
# Prediction Head
#
# At this point, the model has randomly initialized weights.


# ============================================================
# 11. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
# Create the AdamW optimizer.
#
# model.parameters()
# gives all trainable numbers/weights inside the model.
#
# AdamW changes these weights during training.
#
# lr=0.001 controls how large those changes are.
#
# Goal:
# Change the weights so that LOSS becomes smaller.


# ============================================================
# 12. TRAINING LOOP
# ============================================================

for step in range(epochs):
    # Repeat the training process 1500 times.
    #
    # step will be:
    # 0, 1, 2, 3, ... 1499


    xb, yb = get_batch()
    # Get a random batch.
    #
    # xb = input sequences
    # yb = correct next-token sequences
    #
    # Example:
    #
    # xb:
    # [hello friends how are you the]
    #
    # yb:
    # [friends how are you the tea]


    logits, loss = model(xb, yb)
    # Give the input and correct answers to the model.
    #
    # The model:
    #
    # 1. Converts tokens to embeddings
    # 2. Adds positions
    # 3. Runs Transformer blocks
    # 4. Produces logits
    # 5. Compares logits with yb
    # 6. Calculates loss


    optimizer.zero_grad()
    # Clear the gradients from the previous training step.
    #
    # PyTorch normally accumulates gradients.
    # So we clear the old gradients before calculating new ones.


    loss.backward()
    # Backpropagation.
    #
    # Look at the loss and calculate how each trainable
    # parameter contributed to the error.
    #
    # This calculates gradients.


    optimizer.step()
    # Use the gradients to update the model's weights.
    #
    # In simple words:
    #
    # "Change the model slightly so it makes fewer mistakes
    # next time."


    if step % 300 == 0:
        # Every 300 steps, print the training information.
        #
        # % means remainder.
        #
        # 300 % 300 = 0
        # 600 % 300 = 0
        # 900 % 300 = 0


        print(f"Step {step}, loss={loss.item():.4f}")
        # Display the current training step and loss.
        #
        # Example:
        #
        # Step 300, loss=1.8234
        #
        # loss.item() converts the tensor loss into a normal number.
        # :.4f means show 4 digits after the decimal.


# ============================================================
# 13. GIVE THE MODEL STARTING TEXT
# ============================================================

context = torch.tensor(
    [[word2idx["tea"]]],
    dtype=torch.long
)
# Start text generation with the word "hello".
#
# First:
#
# word2idx["hello"]
#
# might give:
#
# 33
#
# So:
#
# context = [[33]]
#
# Shape:
# [1, 1]
#
# 1 sequence
# 1 token


# ============================================================
# 14. GENERATE NEW TEXT
# ============================================================

out = model.generate(
    context,
    max_new_tokens=15
)
# Ask the trained model to generate 15 new tokens.
#
# Starting:
#
# hello
#
# Model might generate:
#
# hello friends how are you the tea is very hot ...
#
# The exact output can change because
# torch.multinomial() samples tokens probabilistically.


# ============================================================
# 15. CONVERT TOKEN IDs BACK TO WORDS
# ============================================================

print("\nGenerated text:\n")
# Print a heading before the generated text.


print(" ".join(idx2word[int(i)] for i in out[0]))
# Convert the generated token IDs back into words.
#
# Example:
#
# out[0]:
# [33, 27, 8, 1]
#
# idx2word:
# 33 → hello
# 27 → friends
# 8  → how
# 1  → are
#
# Result:
#
# hello friends how are