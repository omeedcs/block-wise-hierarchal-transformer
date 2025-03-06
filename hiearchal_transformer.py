from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# FullAttention class with MPS compatibility and improved numerical stability
class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        # queries: [batch_size, n_heads, seq_len_q, d_k]
        # keys: [batch_size, n_heads, seq_len_k, d_k]
        # values: [batch_size, n_heads, seq_len_v, d_v]
        # mask: [batch_size, seq_len] or None
        
        # Compute attention scores with improved numerical stability
        d_k = queries.size(-1)
        scores = torch.matmul(queries / math.sqrt(d_k), keys.transpose(-2, -1))  # Scale before matmul
        
        if mask is not None:
            # Expand mask for broadcasting over heads
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            mask = mask.unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)  # Use finite value instead of -inf
        
        # Apply softmax with improved numerical stability
        scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
        scores_exp = torch.exp(scores - scores_max)
        if mask is not None:
            scores_exp = scores_exp.masked_fill(mask == 0, 0)
        attn = scores_exp / (scores_exp.sum(dim=-1, keepdim=True) + 1e-9)  # Add small epsilon
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # Clip attention weights for stability
        attn = torch.clamp(attn, min=1e-9, max=1.0)
        
        # Apply attention to values
        output = torch.matmul(attn, values)  # [batch_size, n_heads, seq_len_q, d_v]
        return output

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_queries_keys, d_values, n_heads):
        super().__init__()
        self.attention = attention
        self.n_heads = n_heads
        self.d_k = d_queries_keys // n_heads
        self.d_v = d_values // n_heads
        
        # Layer normalization
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_queries_keys)
        self.k_proj = nn.Linear(d_model, d_queries_keys)
        self.v_proj = nn.Linear(d_model, d_values)
        self.o_proj = nn.Linear(d_values, d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1/math.sqrt(2))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, queries, keys, values, mask=None):
        # Handle 4D input for block processing
        orig_shape = queries.shape
        if len(orig_shape) == 4:
            batch, blocks, seq_len, dim = orig_shape
            # Merge batch and blocks dimensions
            queries = queries.reshape(batch * blocks, seq_len, dim)
            keys = keys.reshape(batch * blocks, seq_len, dim)
            values = values.reshape(batch * blocks, seq_len, dim)
            if mask is not None:
                mask = mask.reshape(batch * blocks, seq_len)
        else:
            batch_size = queries.size(0)
            seq_len_q = queries.size(1)
            seq_len_k = keys.size(1)
        
        # Apply layer normalization
        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)
        
        # Project inputs
        q = self.q_proj(queries)  # [batch_size, seq_len_q, d_k * n_heads]
        k = self.k_proj(keys)     # [batch_size, seq_len_k, d_k * n_heads]
        v = self.v_proj(values)   # [batch_size, seq_len_v, d_v * n_heads]
        
        # Reshape for multi-head attention
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        q = q.reshape(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, seq_len_v, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention and dropout
        out = self.attention(q, k, v, mask)  # [batch_size, n_heads, seq_len_q, d_v]
        out = self.dropout(out)
        
        # Reshape and project back
        out = out.transpose(1, 2).reshape(batch_size, seq_len_q, self.n_heads * self.d_v)
        out = self.o_proj(out)
        
        # Final layer norm
        out = self.norm_out(out)
        
        # Restore original shape if needed
        if len(orig_shape) == 4:
            out = out.reshape(batch, blocks, seq_len, dim)
        
        return out

class TransformerLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, n_heads, dropout_ff=0.1, dropout_attn=0.1):
        super().__init__()
        self.attention = AttentionLayer(
            attention=attention,
            d_model=d_model,
            d_queries_keys=d_model,
            d_values=d_model,
            n_heads=n_heads
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout_ff)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_attn)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attended = self.attention(x, x, x, mask)
        x = x + self.dropout(attended)
        
        # Feed-forward with residual
        x = self.feed_forward(x)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.w1.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.w2.weight, gain=1/math.sqrt(2))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # First normalization
        normalized = self.norm1(x)
        
        # First layer
        h = self.w1(normalized)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second layer with residual
        out = self.w2(h)
        out = self.dropout(out)
        
        # Second normalization and residual connection
        out = self.norm2(out + x)
        
        return out

# PoolingLayer for attention-based block aggregation
class PoolingLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Initialize query with small values
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn_layer = AttentionLayer(
            attention=FullAttention(),
            d_model=d_model,
            d_queries_keys=d_model,
            d_values=d_model,
            n_heads=n_heads,
        )

    def forward(self, x, pad_mask=None):
        # x shape: [batch*blocks, block_size, d_model]
        # pad_mask shape: [batch*blocks, block_size] or None
        
        # Apply layer norm to input
        x = self.norm(x)
        
        # Get batch size from input
        batch_size = x.size(0)
        
        # Expand query to match batch size
        query = self.query.expand(batch_size, -1, -1)  # [batch*blocks, 1, d_model]
        
        # Compute attention between query and input sequence
        pooled = self.attn_layer(query, x, x, pad_mask)  # [batch*blocks, 1, d_model]
        
        return pooled.squeeze(1)  # [batch*blocks, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# HierarchicalTransformer class
class HierarchicalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_heads, n_local_layers, n_global_layers, block_size):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Local transformer layers
        self.local_layers = nn.ModuleList([
            TransformerLayer(attention=FullAttention(), d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout_ff=0.1, dropout_attn=0.1)
            for _ in range(n_local_layers)
        ])
        
        # Local output normalization
        self.local_norm = nn.LayerNorm(d_model)
        
        # Pooling layer
        self.pooling = PoolingLayer(d_model=d_model, n_heads=n_heads)
        
        # Global transformer layers
        self.global_layers = nn.ModuleList([
            TransformerLayer(attention=FullAttention(), d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout_ff=0.1, dropout_attn=0.1)
            for _ in range(n_global_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embeddings with small values
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output layer with small values
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Initialize all linear layers with Xavier uniform
        for name, p in self.named_parameters():
            if "linear" in name.lower() or "proj" in name.lower():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=1/math.sqrt(2))
                else:
                    nn.init.zeros_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "norm" in name and "weight" in name:
                nn.init.ones_(p)
            elif "norm" in name and "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1/math.sqrt(2))
    
    def make_attn_mask(self, x, pad_mask):
        batch, length, _ = x.shape
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).expand(batch, -1, -1)
        if pad_mask is not None:
            pad_mask = pad_mask.squeeze(-1).unsqueeze(1).expand(-1, length, -1)
            full_mask = torch.logical_or(pad_mask, causal_mask)
        else:
            full_mask = causal_mask
        return full_mask

    def forward(self, x, pad_mask=None):
        batch_size = x.size(0)
        orig_seq_len = x.size(1)
        
        # Pad sequence to be divisible by block_size
        if orig_seq_len % self.block_size != 0:
            pad_len = self.block_size - (orig_seq_len % self.block_size)
            x = F.pad(x, (0, pad_len), value=0)  # Pad with zeros
            if pad_mask is not None:
                pad_mask = F.pad(pad_mask, (0, pad_len), value=False)
        
        seq_len = x.size(1)
        num_blocks = seq_len // self.block_size
        
        # Embedding with gradient clipping
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = torch.clamp(x, min=-100, max=100)  # Clip embedding values
        
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.input_norm(x)
        
        # Reshape into blocks
        x = x.view(batch_size, num_blocks, self.block_size, -1)  # [batch_size, num_blocks, block_size, d_model]
        
        # Create block-level mask if needed
        if pad_mask is not None:
            block_mask = pad_mask.view(batch_size, num_blocks, self.block_size)
        else:
            block_mask = None
        
        # Process each block with local layers
        for layer in self.local_layers:
            x = layer(x)  # [batch_size, num_blocks, block_size, d_model]
            x = torch.clamp(x, min=-100, max=100)  # Clip after each layer
        
        # Local output normalization
        x = self.local_norm(x)
        
        # Reshape to [batch_size * num_blocks, block_size, d_model] for pooling
        x_flat = x.view(-1, self.block_size, self.d_model)
        if block_mask is not None:
            block_mask_flat = block_mask.view(-1, self.block_size)
        else:
            block_mask_flat = None
        
        # Pool blocks
        pooled = self.pooling(x_flat, block_mask_flat)  # [batch_size * num_blocks, d_model]
        pooled = torch.clamp(pooled, min=-100, max=100)  # Clip pooled values
        
        # Reshape back to [batch_size, num_blocks, d_model]
        pooled = pooled.view(batch_size, num_blocks, -1)
        
        # Process with global layers
        for layer in self.global_layers:
            pooled = layer(pooled)
            pooled = torch.clamp(pooled, min=-100, max=100)  # Clip after each layer
        
        # Output normalization
        pooled = self.output_norm(pooled)
        
        # Project to vocabulary size and reshape with scaled initialization
        logits = self.output_layer(pooled)  # [batch_size, num_blocks, vocab_size]
        logits = torch.clamp(logits, min=-100, max=100)  # Clip logits
        
        logits = logits.view(batch_size, num_blocks, 1, -1)  # Add block dimension
        logits = logits.expand(-1, -1, self.block_size, -1)  # Expand to block size
        logits = logits.contiguous().view(batch_size, seq_len, -1)  # Reshape to sequence length
        
        # Return only the valid sequence length
        if orig_seq_len != seq_len:
            logits = logits[:, :orig_seq_len, :]
        
        return logits  # [batch_size, orig_seq_len, vocab_size]

# Device configuration
device = torch.device("cpu")

# Vocabulary and special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
vocab = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    "hello", "hi", "how", "are", "you", "good", "bye", "what", "is", "your", "name",
    "nice", "to", "meet", "thanks", "see", "later", "today", "i", "am", "too", "omeed",
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
    ["hi what is your name", "hello my name is omeed"],
    ["how are you today", "i am good how about you"],
    ["good thanks", "great to hear bye"],
    ["see you later", "bye"],
    ["hello", "hi there"],
    ["how are you", "good how are you"],
    ["what is your name", "my name is omeed"],
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

# Dataset with padding to MAX_SEQ_LEN
class ChatDataset(Dataset):
    def __init__(self, conversations, max_seq_len=64, pad_token_id=PAD_TOKEN_ID):
        self.conversations = conversations
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        seq = self.conversations[idx]
        if len(seq) < self.max_seq_len:
            seq = seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]
        return torch.tensor(seq, dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch):
    sequences = torch.stack(batch)
    pad_mask = (sequences == PAD_TOKEN_ID).unsqueeze(-1)
    return sequences, pad_mask

# Training function
def train_model(model, dataloader, num_epochs, optimizer, pad_token_id):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (tokens, pad_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(tokens, pad_mask)  # [batch_size, seq_len, vocab_size]
            
            # Ensure shapes match for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            targets = tokens[:, :seq_len].contiguous()
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                print(f"Logits shape: {logits.shape}, Tokens shape: {tokens.shape}")
                print(f"Logits min/max: {logits.min().item()}, {logits.max().item()}")
                print(f"Logits mean/std: {logits.mean().item()}, {logits.std().item()}")
                return
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN/Inf gradient in {name}")
                        has_nan = True
                    grad_norm = param.grad.norm()
                    if grad_norm > 10:
                        print(f"Large gradient norm in {name}: {grad_norm}")
            
            if has_nan:
                return
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

# Hyperparameters
MAX_SEQ_LEN = 64
BLOCK_SIZE = 8
D_MODEL = 128
D_FF = 512
N_HEADS = 4
N_LOCAL_LAYERS = 2
N_GLOBAL_LAYERS = 1
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT_ATTN = 0.1
DROPOUT_RESIDUAL = 0.1
DROPOUT_QKV = 0.1
DROPOUT_EMB = 0.1
WARMUP_STEPS = 100

# Initialize model
model = HierarchicalTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    d_ff=D_FF,
    n_heads=N_HEADS,
    n_local_layers=N_LOCAL_LAYERS,
    n_global_layers=N_GLOBAL_LAYERS,
    block_size=BLOCK_SIZE,
).to(device)

# Use AdamW optimizer with weight decay and gradient clipping
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

dataset = ChatDataset(conversations)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# Train
train_model(model, dataloader, NUM_EPOCHS, optimizer, PAD_TOKEN_ID)
torch.save(model.state_dict(), "hierarchical_chat_transformer.pth")

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
def generate_response(input_text, model, word_to_id, id_to_word, max_length=20, top_k=10, temperature=0.8):
    model.eval()
    tokens = input_text.lower().split()
    input_ids = [SOS_TOKEN_ID] + [word_to_id.get(t, EOS_TOKEN_ID) for t in tokens] + [SEP_TOKEN_ID]
    
    # Pad sequence to be divisible by block_size
    remainder = len(input_ids) % model.block_size
    if remainder > 0:
        pad_len = model.block_size - remainder
        input_ids.extend([PAD_TOKEN_ID] * pad_len)
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).unsqueeze(-1).to(device)
    pad_mask[0, :, 0] = (input_tensor[0] == PAD_TOKEN_ID)

    # Initialize output with just [SOS]
    output_ids = [SOS_TOKEN_ID]
    input_token_set = set(input_ids)  # Track input tokens to avoid repetition
    
    # Keep track of n-grams to prevent repetition
    n_grams = set()
    n = 3  # Use trigrams
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input for next token prediction
            current_ids = input_ids + [SEP_TOKEN_ID] + output_ids[1:]  # Include input context
            remainder = len(current_ids) % model.block_size
            if remainder > 0:
                padding_length = model.block_size - remainder
                current_ids.extend([PAD_TOKEN_ID] * padding_length)
            
            input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
            pad_mask = torch.zeros_like(input_tensor, dtype=torch.bool).unsqueeze(-1).to(device)
            pad_mask[0, :, 0] = (input_tensor[0] == PAD_TOKEN_ID)
            
            # Get model predictions
            logits = model(input_tensor, pad_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for id_ in set(output_ids):
                next_token_logits[0, id_] /= 1.2  # Penalize tokens that have been generated
            for id_ in input_token_set:
                next_token_logits[0, id_] /= 1.5  # Strongly penalize input tokens
                
            # Filter special tokens except EOS
            for id_ in [SOS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID]:
                next_token_logits[0, id_] = float('-inf')
            
            # Check if adding each top-k token would create a repetitive n-gram
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k * 2, next_token_logits.size(-1)), dim=-1)
            valid_indices = []
            for idx in top_k_indices[0]:
                idx = idx.item()
                candidate_token = id_to_word.get(idx, "<UNK>")
                
                # Get the last n-1 tokens
                prev_tokens = [id_to_word.get(tid, "<UNK>") for tid in output_ids[-n+1:]]
                if len(prev_tokens) < n-1:
                    valid_indices.append(idx)
                    continue
                    
                # Check if this would create a repetitive n-gram
                candidate_ngram = " ".join(prev_tokens + [candidate_token])
                if candidate_ngram not in n_grams:
                    valid_indices.append(idx)
                    if len(valid_indices) >= top_k:
                        break
            
            if not valid_indices:  # If all tokens would create repetitive n-grams, use original top-k
                valid_indices = top_k_indices[0, :top_k].tolist()
            
            # Sample from valid tokens
            valid_logits = next_token_logits[0, valid_indices]
            probs = F.softmax(valid_logits, dim=-1)
            next_token_id = valid_indices[torch.multinomial(probs, 1).item()]
            
            output_ids.append(next_token_id)
            
            # Update n-grams
            if len(output_ids) >= n:
                current_tokens = [id_to_word.get(tid, "<UNK>") for tid in output_ids[-n:]]
                n_grams.add(" ".join(current_tokens))
            
            if next_token_id == EOS_TOKEN_ID:
                break
    
    # Convert tokens to words, skipping special tokens
    output_tokens = []
    for token_id in output_ids:
        if token_id == EOS_TOKEN_ID:
            break
        if token_id not in [SOS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID]:
            output_tokens.append(id_to_word[token_id])
    
    response = " ".join(output_tokens)
    return response if response.strip() else "I am doing well, thank you! How can I help you today?"

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