#!/usr/bin/env python
# coding: utf-8

# In[26]:


# import ipynbname

data_dir = "C:/Users/16072/Desktop/DeepLearning/gpt-dev/data"
# read it in to inspect it
with open(data_dir + "/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# In[27]:


print("length of dataset in characters: ", len(text))


# In[28]:


# let's look at the first 1000 characters
print(text[:1000])


# In[29]:


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)


# In[30]:


# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}  # character to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to character
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))


# In[31]:


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])  # first 1000 characters encoded as integers


# In[32]:


train_ratio = 0.9
n = int(train_ratio * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# In[33]:


block_size = 8
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context.tolist()} the target: {target}")


# In[34]:


torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb)
print("targets:")
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")


# In[35]:


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

n_embd = 32


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd // 4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.sa_head(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            # logits, _ = self(idx)
            # print(idx, logits.shape)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # append sampled index to the running sequence
        return idx


model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

eval_iters = 200


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# In[36]:


idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(idx.shape)
# print(decode(m.generate(idx = idx, max_new_tokens=100)[0].tolist()))


# In[37]:


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# In[38]:


batch_size = 32
max_iters = 5000

for steps in range(max_iters):

    if steps % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


# In[39]:


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


# In[40]:


torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape


# In[41]:


xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)
print(x[0])


# In[42]:


xbow[0]


# In[43]:


# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a=")
print(a)
print("--")
print("b=")
print(b)
print("--")
print("c=")
print(c)


# In[44]:


tril = torch.tril(torch.ones((block_size, block_size)))
wei = torch.zeros((block_size, block_size))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(xbow3[0])


# In[45]:


# version 3: self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B,T,head_size)
q = query(x)  # (B,T,head_size)
wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B,T,T)

tril = torch.tril(torch.ones((T, T)))
# wei = torch.zeros((block_size, block_size))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

v = value(x)  # (B,T,head_size)
out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
# xbow3 = wei @ x
print(out.shape)
print(out)


# In[46]:


X, Y = get_batch("train")
print(X.shape, X[0])
print(Y.shape, Y[0])


# In[49]:


import inspect

print(inspect.getsourcefile(BigramLanguageModel.forward))
print(BigramLanguageModel.forward.__code__.co_firstlineno)
print(inspect.getsource(BigramLanguageModel.forward))


# In[50]:


import ipdb

ipdb.set_trace()
m.forward(X, Y)


# In[ ]:
