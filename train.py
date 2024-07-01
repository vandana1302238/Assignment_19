import requests
import wget
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel


# hyperparameters
batch_size = 64 # no. of independent sequences processes in parallel
block_size = 256 #length of random data size to be trained. (possibly also called context length)
max_iters = 4500
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50

#---------

torch.manual_seed(1337)


# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

# Save the file locally
with open('input.txt', 'wb') as file:
    file.write(response.content)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(text[:500])

#all unique chars in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

#Tokenize
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))

#encode full dataset
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

#split training and validation data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# train_data[:block_size+1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
#
# print('-----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")





# model = BigramLanguageModel(vocab_size)
torch.cuda.empty_cache()
torch.cuda.amp.autocast(enabled = True)

model = BigramLanguageModel(vocab_size, block_size, device)
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M Parameters')

# logits, loss = (xb, yb)
# print(logits.shape)
# print(loss)

# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()
for iter in range(max_iters):
    torch.cuda.empty_cache()
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)  # m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    # loss.backward()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    #optimizer.step()

# print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), "S19BasicGPT_model.pth")