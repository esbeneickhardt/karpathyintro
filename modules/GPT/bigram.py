#################
### Libraries ###
#################

import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available else 'cpu'; device = 'cpu'

############
### Data ###
############

# Reading Data
with open("../../data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# All unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Building the vocabulary
ctoi = {s:i for i,s in enumerate(chars)}
itoc = {i:s for s,i in ctoi.items()}

# Building encoder/decoder
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

# Tokenizing dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train/Valid Split
n = int(0.9*len(data))
data_train = data[:n]
data_valid = data[n:]

# Gets Ramdom Batches
def get_batch(split: str, batch_size: int, block_size: int) -> torch.tensor:
    """
    Description:
        Generates a batch of data of inputs x and targets y.
    Inputs:
        split: test or valid split
        batch_size: How many independent sequences will be processed in parallel
        block_size: Maximum context length
    Outputs:
        x, y: a tuple with xs and ys
    """
    data = data_train if split == 'train' else data_valid
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

#############
### Model ###
#############

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        """ Creating Embedding Table """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx: torch.tensor, targets: torch.tensor=None) -> tuple:
        """ Calculating the Loss """
        logits = self.token_embedding_table(idx) # (BATCH, TIME, CHANNEL)
        if targets is None:
            loss = None
        else:
            # Loss function takes (BATCH, CHANNEL, TIME) so we rearrange
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        """ Generates Tokens Using a Sliding Window """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss():
    """
    Description:
        Estimates losses on train and valid
    Outputs:
        out: Mean loss across eval_iters items
    """
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Creating model
model = BigramLanguageModel(vocab_size)
model = model.to(device)

################
### Training ###
################

# Hyperparameters
batch_size = 32
block_size = 8
max_steps = 10000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

# Creating PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_steps):
    
    # Once in a while evaluate loss on train and valid sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train', batch_size, block_size)
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

##################
### Generating ###
##################

# Generating some text
model_input = torch.zeros((1,1), dtype=torch.long) # Input token 0, which is \n
model_output = model.generate(model_input, max_new_tokens=300)[0].tolist()
print(f"Generated Text: \n {decode(model_output)}")