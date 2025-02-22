from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange

# Normalization class
class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)

# FullAttention class with fix for MPS compatibility
class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L, S]
            scores = scores.masked_fill(attn_mask, -torch.inf)
        A = torch.softmax(scale * scores, dim=-1)  # [B, H, L, S]
        A = self.dropout(A)
        if attn_mask is not None:
            A = A.masked_fill(attn_mask, 0.0)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V, A

# AttentionLayer class
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_queries_keys, d_values, n_heads, dropout_qkv=0.0):
        super().__init__()
        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)
        out, attn = self.attention(queries=queries, keys=keys, values=values, attn_mask=attn_mask)
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out, attn

# TransformerLayer class
class TransformerLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, dropout_ff=0.1, activation="gelu", norm="layer"):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, self_seq, self_mask=None, cross_seq=None, cross_mask=None):
        q1 = self.norm1(self_seq)
        q1, self_attn = self.self_attention(queries=q1, keys=q1, values=q1, attn_mask=self_mask)
        self_seq = self_seq + q1
        if self.cross_attention is not None:
            q1 = self.norm2(self_seq)
            q1, cross_attn = self.cross_attention(queries=q1, keys=cross_seq, values=cross_seq, attn_mask=cross_mask)
            self_seq = self_seq + q1
        else:
            cross_attn = None
        q1 = self.norm3(self_seq)
        q1 = self.dropout_ff(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq, {"self_attn": self_attn, "cross_attn": cross_attn}

# Transformer class for chat-based tasks
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 200,
        d_ff: int = 600,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.0,
        dropout_qkv: float = 0.05,
        activation: str = "gelu",
        norm: str = "layer",
    ):
        super().__init__()
        assert activation in ["gelu", "relu"]
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_emb)

        def make_attn():
            return AttentionLayer(
                attention=FullAttention(attention_dropout=dropout_attn),
                d_model=d_model,
                d_queries_keys=d_model // n_heads,
                d_values=d_model // n_heads,
                n_heads=n_heads,
                dropout_qkv=dropout_qkv,
            )

        def make_layer():
            return TransformerLayer(
                self_attention=make_attn(),
                cross_attention=None,
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def make_attn_mask(self, x, pad_mask):
        batch, length, _ = x.shape
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).expand(batch, -1, -1)
        pad_mask = pad_mask.squeeze(-1).unsqueeze(1).expand(-1, length, -1)
        full_mask = torch.logical_or(pad_mask, causal_mask)
        return full_mask

    def forward(self, tokens, pad_mask):
        batch, length = tokens.shape
        pos_idxs = torch.arange(length, device=tokens.device)
        token_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(pos_idxs)
        emb = token_emb + pos_emb
        emb = self.dropout(emb)
        mask = self.make_attn_mask(emb, pad_mask)
        for layer in self.layers:
            emb, _ = layer(self_seq=emb, self_mask=mask)
        emb = self.norm(emb)
        logits = self.output_layer(emb)
        return logits

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Vocabulary and special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
vocab = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    "hello", "hi", "how", "are", "you", "good", "bye", "what", "is", "your", "name",
    "nice", "to", "meet", "thanks", "see", "later", "today", "i", "am", "too", "grok",
    "doing", "well", "favorite", "color", "blue", "music", "enjoy", "weather", "sunny",
    "old", "timeless", "joke", "chicken", "road", "side", "purpose", "assist", "chat",
    "dream", "imagine", "food", "eat", "suggest", "recipes", "feeling", "helpful", "there",
    "about", "great", "hear"
]
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for idx, word in enumerate(vocab)}
PAD_TOKEN_ID = word_to_id[PAD_TOKEN]
SOS_TOKEN_ID = word_to_id[SOS_TOKEN]
EOS_TOKEN_ID = word_to_id[EOS_TOKEN]
SEP_TOKEN_ID = word_to_id[SEP_TOKEN]
VOCAB_SIZE = len(vocab)

# Expanded training data
raw_conversations = [
    ["hello how are you", "hi i am good thanks"],
    ["nice to meet you", "nice to meet you too"],
    ["hi what is your name", "hello my name is grok"],
    ["how are you today", "i am good how about you"],
    ["good thanks", "great to hear bye"],
    ["see you later", "bye"],
    ["hello", "hi there"],
    ["how are you", "good how are you"],
    ["what is your name", "my name is grok"],
    ["nice to meet you", "thanks nice to meet you too"],
    ["hello how are you doing", "hi i am doing well thanks"],
    ["what is your favorite color", "i like blue"],
    ["do you like music", "yes i enjoy music"],
    ["what is the weather like", "it is sunny today"],
    ["how old are you", "i am timeless"],
    ["tell me a joke", "why did the chicken cross the road to get to the other side"],
    ["what is your purpose", "to assist and chat with you"],
    ["do you dream", "i do not dream but i can imagine"],
    ["what is your favorite food", "i do not eat but i can suggest recipes"],
    ["how are you feeling", "i am feeling helpful today"],
]

def tokenize_conversation(conversation, word_to_id, sos_token_id, eos_token_id, sep_token_id):
    tokenized_pairs = []
    for i in range(len(conversation) - 1):
        input_tokens = [sos_token_id] + [word_to_id.get(token, eos_token_id) for token in conversation[i].split()] + [sep_token_id]
        response_tokens = [word_to_id.get(token, eos_token_id) for token in conversation[i+1].split()] + [eos_token_id]
        tokenized_pairs.append(input_tokens + response_tokens)
    return tokenized_pairs

conversations = []
for conv in raw_conversations:
    conversations.extend(tokenize_conversation(conv, word_to_id, SOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID))

# Dataset
class ChatDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return torch.tensor(self.conversations[idx], dtype=torch.long)

def collate_fn(batch, pad_token_id):
    sequences = [seq for seq in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    pad_mask = (padded_sequences == pad_token_id).unsqueeze(-1)
    return padded_sequences, pad_mask

# Training function with debugging
def train_model(model, dataloader, num_epochs, optimizer, pad_token_id):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (tokens, pad_mask) in enumerate(dataloader):
            tokens = tokens.to(device)
            pad_mask = pad_mask.to(device)
            logits = model(tokens, pad_mask)
            target = tokens[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_token_id
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

# Hyperparameters
MAX_SEQ_LEN = 64
D_MODEL = 256
D_FF = 1024
N_HEADS = 8
LAYERS = 6
DROPOUT_EMB = 0.1
DROPOUT_FF = 0.1
DROPOUT_ATTN = 0.0
DROPOUT_QKV = 0.1
ACTIVATION = "gelu"
NORM = "layer"
NUM_EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# Initialize model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    d_model=D_MODEL,
    d_ff=D_FF,
    n_heads=N_HEADS,
    layers=LAYERS,
    dropout_emb=DROPOUT_EMB,
    dropout_ff=DROPOUT_FF,
    dropout_attn=DROPOUT_ATTN,
    dropout_qkv=DROPOUT_QKV,
    activation=ACTIVATION,
    norm=NORM,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = ChatDataset(conversations)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, PAD_TOKEN_ID)
)

# Train
train_model(model, dataloader, NUM_EPOCHS, optimizer, PAD_TOKEN_ID)
torch.save(model.state_dict(), "chat_transformer.pth")

# Perplexity evaluation
def calculate_perplexity(model, dataloader, pad_token_id):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for tokens, pad_mask in dataloader:
            tokens = tokens.to(device)
            pad_mask = pad_mask.to(device)
            logits = model(tokens, pad_mask)
            target = tokens[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_token_id,
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += (target != pad_token_id).sum().item()
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

perplexity = calculate_perplexity(model, dataloader, PAD_TOKEN_ID)
print(f"Final Perplexity: {perplexity:.4f}")

# Evaluation function with top-k sampling
def generate_response(input_text, model, word_to_id, id_to_word, max_length=20, top_k=10):
    model.eval()
    tokens = input_text.lower().split()
    input_ids = [SOS_TOKEN_ID] + [word_to_id.get(t, EOS_TOKEN_ID) for t in tokens] + [SEP_TOKEN_ID]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).unsqueeze(-1).to(device)

    output_ids = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor, pad_mask)
            next_token_logits = logits[:, -1, :]
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            next_token_id = top_k_indices[0, torch.multinomial(top_k_probs, 1).item()].item()
            output_ids.append(next_token_id)
            if next_token_id == EOS_TOKEN_ID:
                break
            input_tensor = torch.tensor([output_ids], dtype=torch.long).to(device)
            pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).unsqueeze(-1).to(device)

    sep_idx = output_ids.index(SEP_TOKEN_ID) if SEP_TOKEN_ID in output_ids else -1
    response_tokens = [id_to_word.get(idx, "<UNK>") for idx in output_ids[sep_idx+1:]]
    return " ".join(response_tokens)

# Test the model
test_inputs = [
    "hello how are you",
    "hi what is your name",
    "nice to meet you",
    "how are you today"
]

for input_text in test_inputs:
    response = generate_response(input_text, model, word_to_id, id_to_word)
    print(f"User Input: {input_text}")
    print(f"Model Response: {response}")
    print("---")