from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange
import logging
import math
import os
import random
from collections import Counter, defaultdict
import re
import json
from bpe_tokenizer import BPETokenizer, load_or_create_bpe_tokenizer

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
    def __init__(self, vocab_size, d_model=512, d_ff=2048, n_heads=8, n_local_layers=4, n_global_layers=2, block_size=8):
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
        """Initialize model weights with improved initialization strategy"""
        # Embedding initialization
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Output layer initialization
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Apply layernorm initialization
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
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

# Vocabulary and special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = ""
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
UNK_TOKEN = "<UNK>"  # Add unknown token

def load_tokenizer_and_vocab(texts=None, vocab_size=2000, tokenizer_path="bpe_tokenizer.json"):
    """Load or create BPE tokenizer and vocabulary mappings"""
    special_tokens = {
        'PAD': PAD_TOKEN,
        'SOS': SOS_TOKEN,
        'EOS': EOS_TOKEN,
        'SEP': SEP_TOKEN,
        'UNK': UNK_TOKEN
    }
    
    # If no texts are provided, load from synthetic data or existing conversations
    if texts is None:
        try:
            # Try to load from enhanced_conversations.json
            if os.path.exists("enhanced_conversations.json"):
                print("Loading texts from enhanced_conversations.json for tokenizer training")
                with open("enhanced_conversations.json", "r") as f:
                    conversations = json.load(f)
                texts = []
                # The file format is a list of conversations, where each conversation is a list of alternating messages
                for conversation in conversations:
                    if isinstance(conversation, list):
                        for message in conversation:
                            if isinstance(message, str):
                                texts.append(message)
            # If that doesn't exist, generate some synthetic data
            else:
                print("No existing conversation data found. Generating synthetic data for tokenizer training...")
                # Import here to avoid circular imports
                from data_loader import DataProcessor, SyntheticDataSource, get_conversation_templates
                data_processor = DataProcessor()
                templates = get_conversation_templates()
                synthetic_source = SyntheticDataSource(templates, count=500)  # Generate 500 samples for tokenizer training
                data_processor.add_source(synthetic_source)
                conversations = data_processor.process_all_sources()
                texts = []
                for conv in conversations:
                    for turn in conv:
                        if isinstance(turn, dict) and "text" in turn:
                            texts.append(turn["text"])
                        elif isinstance(turn, str):
                            texts.append(turn)
                # Save for future use
                with open("tokenizer_training_data.json", "w") as f:
                    json.dump(conversations, f)
        except Exception as e:
            print(f"Error loading texts for tokenizer training: {e}")
            raise ValueError(f"Failed to load or generate texts for tokenizer training: {e}")
            
        if not texts:
            raise ValueError("No texts could be loaded for tokenizer training")
    
    # Load or create BPE tokenizer
    tokenizer = load_or_create_bpe_tokenizer(
        texts=texts, 
        vocab_size=vocab_size,
        save_path=tokenizer_path,
        special_tokens=special_tokens
    )
    
    # Get token IDs for special tokens
    PAD_TOKEN_ID = tokenizer.get_special_token_id('PAD')
    SOS_TOKEN_ID = tokenizer.get_special_token_id('SOS')
    EOS_TOKEN_ID = tokenizer.get_special_token_id('EOS')
    SEP_TOKEN_ID = tokenizer.get_special_token_id('SEP')
    UNK_TOKEN_ID = tokenizer.get_special_token_id('UNK')
    
    # Create word-to-id and id-to-word mappings
    word_to_id = tokenizer.encoder
    id_to_word = tokenizer.decoder
    
    # Create word-to-id and id-to-word mappings as string keys/values for backward compatibility
    word_to_id_str = {str(k): v for k, v in word_to_id.items()}
    id_to_word_str = {str(k): v for k, v in id_to_word.items()}
    
    vocab_size = len(tokenizer.encoder)
    logger.info(f"Vocabulary size: {vocab_size} tokens")
    
    return tokenizer, word_to_id, id_to_word, vocab_size, {
        'PAD_TOKEN_ID': PAD_TOKEN_ID,
        'SOS_TOKEN_ID': SOS_TOKEN_ID,
        'EOS_TOKEN_ID': EOS_TOKEN_ID,
        'SEP_TOKEN_ID': SEP_TOKEN_ID,
        'UNK_TOKEN_ID': UNK_TOKEN_ID
    }

# Tokenize text using BPE tokenizer
def tokenize(text, tokenizer):
    """Convert text string to token IDs using BPE tokenizer"""
    return tokenizer.tokenize(text)

# Detokenize token IDs
def detokenize(token_ids, tokenizer):
    """Convert token IDs to text string using BPE tokenizer"""
    return tokenizer.detokenize(token_ids)

# Tokenize a conversation
def tokenize_conversation(conversation, tokenizer, special_token_ids):
    """
    Tokenize a conversation into training pairs
    Returns list of (input_ids, target_ids) pairs
    """
    sos_token_id = special_token_ids['SOS_TOKEN_ID']
    eos_token_id = special_token_ids['EOS_TOKEN_ID']
    sep_token_id = special_token_ids['SEP_TOKEN_ID']
    
    results = []
    for i in range(0, len(conversation)-1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i+1] if i+1 < len(conversation) else ""
        
        # Skip empty turns
        if not user_turn or not assistant_turn:
            continue
            
        # Tokenize user input and assistant response
        user_ids = tokenizer.tokenize(user_turn.lower())
        assistant_ids = tokenizer.tokenize(assistant_turn.lower())
        
        # Create training pair - format: [SOS] user [SEP] assistant [EOS]
        input_ids = [sos_token_id] + user_ids + [sep_token_id]
        target_ids = input_ids + assistant_ids + [eos_token_id]
        
        results.append((input_ids, target_ids))
        
    return results

# Load tokenizer and create vocab
tokenizer, word_to_id, id_to_word, VOCAB_SIZE, special_token_ids = load_tokenizer_and_vocab()

# Extract special token IDs for convenience
PAD_TOKEN_ID = special_token_ids['PAD_TOKEN_ID']
SOS_TOKEN_ID = special_token_ids['SOS_TOKEN_ID']
EOS_TOKEN_ID = special_token_ids['EOS_TOKEN_ID']
SEP_TOKEN_ID = special_token_ids['SEP_TOKEN_ID']
UNK_TOKEN_ID = special_token_ids['UNK_TOKEN_ID']

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

conversations = []
for conv in raw_conversations:
    conversations.extend(tokenize_conversation(conv, tokenizer, special_token_ids))

# Dataset with padding to MAX_SEQ_LEN
class ChatDataset(Dataset):
    def __init__(self, conversations, max_seq_len=64, pad_token_id=special_token_ids['PAD_TOKEN_ID']):
        self.conversations = conversations
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        seq = self.conversations[idx]
        if not isinstance(seq, list):
            # Handle case where conversation is not already a list
            seq = list(seq)
        
        # Split into input and target (for training)
        # Find separator token position
        try:
            sep_idx = seq.index(special_token_ids['SEP_TOKEN_ID'])
        except ValueError:
            # If no separator, treat entire sequence as input
            sep_idx = len(seq) // 2
        
        # Input is everything up to and including SEP_TOKEN
        input_seq = seq[:sep_idx+1]
        # Target is everything after SEP_TOKEN
        target_seq = seq[sep_idx+1:]
        
        # Pad sequences if needed
        if len(input_seq) < self.max_seq_len:
            input_seq = input_seq + [self.pad_token_id] * (self.max_seq_len - len(input_seq))
        else:
            input_seq = input_seq[:self.max_seq_len]
            
        if len(target_seq) < self.max_seq_len:
            target_seq = target_seq + [self.pad_token_id] * (self.max_seq_len - len(target_seq))
        else:
            target_seq = target_seq[:self.max_seq_len]
        
        # Return input, target sequences
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )

# Collate function for DataLoader
def collate_fn(batch, pad_token_id=special_token_ids['PAD_TOKEN_ID']):
    # Separate input and target sequences
    input_seqs, target_seqs = zip(*batch)
    
    # Stack into tensors
    input_tensor = torch.stack(input_seqs)
    target_tensor = torch.stack(target_seqs)
    
    # Create padding mask for inputs
    pad_mask = (input_tensor == pad_token_id).unsqueeze(-1)
    
    return input_tensor, target_tensor, pad_mask

# Training function
def train_model(model, dataloader, num_epochs, optimizer, pad_token_id):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_tensor, target_tensor, pad_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_tensor, pad_mask)  # [batch_size, seq_len, vocab_size]
            
            # Ensure shapes match for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            targets = target_tensor[:, :seq_len].contiguous()
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                print(f"Logits shape: {logits.shape}, Targets shape: {target_tensor.shape}")
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

# Initialize the model with default parameters
if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    
    # Create the model
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

    # Create a simple dataset for demonstration
    from data_loader import DataProcessor, SyntheticDataSource, get_conversation_templates
    data_processor = DataProcessor()
    templates = get_conversation_templates()
    synthetic_source = SyntheticDataSource(templates, count=100)
    data_processor.add_source(synthetic_source)
    conversations = data_processor.process_all_sources()

    dataset = ChatDataset(conversations)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Calculate initial perplexity
    perplexity = calculate_perplexity(model, dataloader, special_token_ids['PAD_TOKEN_ID'])
    print(f"Initial Perplexity: {perplexity:.4f}")

    print("Training model with default settings...")
    print("For advanced training options, use enhanced_training.py instead.")
    import sys
    import subprocess
    
    # Forward to enhanced training with default parameters
    command = [sys.executable, "enhanced_training.py"]
    subprocess.run(command)

    print("\nTesting model with example conversations...")
    test_inputs = [
        "hello",
        "what is the weather today",
        "tell me a joke",
        "how are you today"
    ]
    
    for input_text in test_inputs:
        response = generate_response(model, input_text, tokenizer, device)
        print(f"User Input: {input_text}")
        print(f"Model Response: {response}")
        print("---")

# Perplexity evaluation
def calculate_perplexity(model, dataloader, pad_token_id):
    model.eval()
    total_loss = 0
    total_tokens = 0
    # Get the device from the model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for input_tensor, target_tensor, pad_mask in dataloader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            pad_mask = pad_mask.to(device)
            logits = model(input_tensor, pad_mask)
            target = target_tensor[:, 1:].contiguous()
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

# Evaluation function with top-k sampling and improved handling of repetitions
def generate_response(model, input_text, tokenizer, device, max_length=40, temperature=0.7, 
                      top_k=30, repetition_penalty=1.5, use_beam_search=True, beam_width=3, 
                      length_normalization_alpha=0.6):
    """
    Generate a response from the model given input text
    
    Args:
        model: The hierarchical transformer model
        input_text: Text input from user
        tokenizer: BPE tokenizer
        device: Device to run generation on
        max_length: Maximum response length
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of highest probability tokens to consider for sampling
        repetition_penalty: Penalty for repeating tokens (higher = less repetition)
        use_beam_search: Whether to use beam search for generation
        beam_width: Number of beams for beam search
        length_normalization_alpha: Parameter for length normalization (0-1)
        
    Returns:
        Generated response text
    """
    model.eval()  # Set model to evaluation mode
    
    # Tokenize the input text
    input_ids = tokenize(input_text, tokenizer)
    
    # Add SOS token at the beginning
    input_ids = [special_token_ids['SOS_TOKEN_ID']] + input_ids
    
    # Convert to tensor and move to device
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Create pad mask (no padding in input)
    pad_mask = torch.ones_like(input_tensor).bool().to(device)
    
    if use_beam_search:
        # Implement beam search 
        sequences = [([special_token_ids['SOS_TOKEN_ID']] + input_ids, 0.0)]  # (sequence, score)
        finalized_sequences = []
        
        for _ in range(max_length):
            all_candidates = []
            
            # Expand each candidate sequence
            for seq, score in sequences:
                if seq[-1] == special_token_ids['EOS_TOKEN_ID'] or seq[-1] == special_token_ids['SEP_TOKEN_ID']:  # Sequence already complete
                    finalized_sequences.append((seq, score))
                    continue
                
                input_seq = torch.tensor(seq).unsqueeze(0).to(device)
                pad_mask = torch.ones_like(input_seq).bool().to(device)
                
                with torch.no_grad():
                    logits = model(input_seq, pad_mask)
                    
                next_token_logits = logits[0, -1, :].cpu()
                
                # Apply repetition penalty
                for token_id in set(seq[len(input_ids):]):  # Only penalize tokens in the response
                    count = seq.count(token_id)
                    if count > 1:
                        next_token_logits[token_id] /= (repetition_penalty ** count)
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits.fill_(-float('inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=0)
                
                # Get the top beam_width candidates
                topk_probs, topk_indices = torch.topk(probs, beam_width)
                
                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_prob = topk_probs[i].item()
                    
                    # Create new sequence
                    new_seq = seq + [token_id]
                    
                    # Apply length normalization to score
                    # (len(new_seq) - len(input_ids)) is the length of just the response part
                    response_len = len(new_seq) - len(input_ids)
                    if length_normalization_alpha > 0 and response_len > 0:
                        length_penalty = ((5 + response_len) / 6) ** length_normalization_alpha
                    else:
                        length_penalty = 1.0
                    
                    # Score is log probability
                    new_score = score + math.log(token_prob) / length_penalty
                    
                    all_candidates.append((new_seq, new_score))
            
            # Handle case where all sequences are finalized
            if len(all_candidates) == 0:
                break
                
            # Sort candidates by score and select top beam_width
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = all_candidates[:beam_width]
            
            # If all beams have ended, break
            if all(seq[-1] == special_token_ids['EOS_TOKEN_ID'] or seq[-1] == special_token_ids['SEP_TOKEN_ID'] for seq, _ in sequences):
                finalized_sequences.extend(sequences)
                break
        
        # Select sequence with highest score
        if finalized_sequences:
            finalized_sequences.sort(key=lambda x: x[1], reverse=True)
            best_seq = finalized_sequences[0][0]
        else:
            best_seq = sequences[0][0]
        
        # Extract only the response part (after input tokens)
        output_token_ids = best_seq[len(input_ids):]
        
        # Remove any EOS/SEP token at the end
        if output_token_ids and (output_token_ids[-1] == special_token_ids['EOS_TOKEN_ID'] or output_token_ids[-1] == special_token_ids['SEP_TOKEN_ID']):
            output_token_ids = output_token_ids[:-1]
    else:
        # Regular autoregressive generation (original implementation)
        output_token_ids = []
        token_counts = {}
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output
                logits = model(input_tensor, pad_mask)
                
                # Consider only the prediction for the last token
                next_token_logits = logits[0, -1, :].cpu()
                
                # Apply repetition penalty
                for token_id in set(output_token_ids):
                    if token_id in token_counts:
                        token_counts[token_id] += 1
                    else:
                        token_counts[token_id] = 1
                    
                    # Apply exponential penalty based on frequency
                    penalty = repetition_penalty ** token_counts[token_id]
                    next_token_logits[token_id] /= penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    
                    # Create a mask for the top-k tokens
                    next_token_logits.fill_(-float('inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=0)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, 1).item()
                
                # Break if we generate the SEP or END token
                if next_token == special_token_ids['SEP_TOKEN_ID'] or next_token == special_token_ids['EOS_TOKEN_ID']:
                    break
                    
                # Add the generated token to our output
                output_token_ids.append(next_token)
                
                # Append the next token to the input for the next iteration
                next_token_tensor = torch.tensor([[next_token]]).to(device)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                
                # Update pad mask
                pad_mask = torch.cat([pad_mask, torch.ones_like(next_token_tensor).bool().to(device)], dim=1)
    
    # Convert token ids back to words
    output_words = []
    for token_id in output_token_ids:
        if token_id in id_to_word:
            output_words.append(id_to_word[token_id])
        else:
            output_words.append("<UNK>")  # Should not happen, but just in case
    
    # Enhanced coherence check
    unique_words = set(output_words)
    
    # Check various criteria for a good response
    min_length = 4
    min_unique_words = 4
    max_repetition_ratio = 0.25  # No word should represent more than 25% of the response
    
    # Semantic checks
    has_content_words = any(word not in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at'] 
                           for word in output_words)
    
    # Check if the response meets our criteria
    if (len(output_words) < min_length or  # Minimum length check
        len(unique_words) < min_unique_words or  # Minimum unique words
        not has_content_words or  # Must have some content words
        any(output_words.count(word) > len(output_words) * max_repetition_ratio for word in unique_words)):  # Repetition check
        return "I'm not sure how to respond to that. Could you rephrase your question?"
    
    # Join the words to create the response text
    response_text = " ".join(output_words)
    
    return response_text