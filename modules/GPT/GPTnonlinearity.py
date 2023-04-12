#################
### Libraries ###
#################
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available else 'cpu'; device = 'cpu'

#######################
### Hyperparameters ###
#######################
batch_size = 32
block_size = 8
max_steps = 10000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embed = 32 

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
class Head(nn.Module):
    """ One head of self-attention """
    
    def __init__(self, head_size):
        """ Creating three linear layers and a mask """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        """ Attention calculation """
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 ## **-0.5 normalize variance to 1
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        """ Multiple heads in parallel """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
    def forward(self, x):
        """ Calculating and Concatenating Results """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out

class FeedForward(nn.Module):
    """ A linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        """ Creating Layers """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.ffw = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx: torch.tensor, targets: torch.tensor=None) -> tuple:
        """ Calculating the Loss """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,embed_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        x = self.ffw(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
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
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Function for estimating loss
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
model = BigramLanguageModel()
model = model.to(device)

################
### Training ###
################

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