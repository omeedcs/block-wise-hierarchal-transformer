"""
Byte Pair Encoding (BPE) Tokenizer Implementation
This implementation provides subword tokenization to improve vocabulary efficiency
"""

import re
import json
import os
from collections import Counter, defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer for subword tokenization
    """
    def __init__(self, vocab_size=2000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.encoder = {}  # token -> id
        self.decoder = {}  # id -> token
        self.special_tokens = {}  # special token name -> id
        self.merges = {}  # Pairs to merge
        self.pattern = re.compile(r'\s+|[^\s\w]+|\w+')  # Tokenization pattern
        
    def add_special_tokens(self, special_tokens):
        """
        Add special tokens to the vocabulary

        Args:
            special_tokens: Dictionary mapping token names to token strings
        """
        token_strings = list(special_tokens.values())
        # Add special tokens to encoder and decoder
        for i, (name, token) in enumerate(special_tokens.items()):
            if token not in self.encoder:
                token_id = len(self.encoder)
                self.encoder[token] = token_id
                self.decoder[token_id] = token
                self.special_tokens[name] = token_id
        
        logger.info(f"Added {len(special_tokens)} special tokens: {', '.join(token_strings)}")
    
    def get_special_token_id(self, name):
        """Get ID for a special token by name"""
        return self.special_tokens.get(name)
    
    def train(self, texts, save_path=None):
        """
        Train the BPE tokenizer on a corpus of texts

        Args:
            texts: List of text strings
            save_path: Optional path to save the vocabulary and merges
        """
        logger.info(f"Training BPE tokenizer on {len(texts)} texts to vocab size {self.vocab_size}")
        
        # Step 1: Collect initial vocabulary (character-level)
        word_counts = Counter()
        for text in texts:
            tokens = re.findall(self.pattern, text.lower())
            word_counts.update(tokens)
        
        # Filter by frequency
        word_counts = {w: c for w, c in word_counts.items() if c >= self.min_frequency}
        
        # Step 2: Split all words into characters and add end-of-word token
        vocab = set()
        word_pieces = {}
        for word, count in word_counts.items():
            chars = list(word)
            word_pieces[word] = chars
            vocab.update(chars)
        
        # Create initial vocabulary (character-level)
        # First, preserve existing special tokens
        base_vocab = list(self.encoder.keys())
        
        # Add characters not yet in the vocabulary
        for char in sorted(vocab):
            if char not in self.encoder:
                token_id = len(self.encoder)
                self.encoder[char] = token_id
                self.decoder[token_id] = char
        
        # Calculate initial merges
        base_vocab_size = len(self.encoder)
        merges_to_learn = self.vocab_size - base_vocab_size
        
        if merges_to_learn <= 0:
            logger.warning(f"Vocabulary already at target size. No merges needed.")
            return
        
        logger.info(f"Learning {merges_to_learn} merges from initial vocab size {base_vocab_size}")
        
        # Step 3: Iteratively merge the most frequent pairs
        merges = []
        for _ in range(merges_to_learn):
            # Count pair frequencies
            pair_counts = Counter()
            for word, count in word_counts.items():
                pieces = word_pieces[word]
                if len(pieces) == 1:
                    continue
                for i in range(len(pieces) - 1):
                    pair = (pieces[i], pieces[i+1])
                    pair_counts[pair] += count
            
            if not pair_counts:
                break
            
            # Find the most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            merges.append(best_pair)
            
            # Create new token for the pair
            new_token = ''.join(best_pair)
            if new_token not in self.encoder:
                token_id = len(self.encoder)
                self.encoder[new_token] = token_id
                self.decoder[token_id] = new_token
            
            # Update word pieces
            for word in word_counts.keys():
                pieces = word_pieces[word]
                new_pieces = []
                i = 0
                while i < len(pieces):
                    if i < len(pieces) - 1 and (pieces[i], pieces[i+1]) == best_pair:
                        new_pieces.append(new_token)
                        i += 2
                    else:
                        new_pieces.append(pieces[i])
                        i += 1
                word_pieces[word] = new_pieces
        
        # Store merges
        self.merges = {f"{x}+{y}": ''.join([x, y]) for x, y in merges}
        
        logger.info(f"Final vocabulary size: {len(self.encoder)} tokens")
        
        if save_path:
            self.save(save_path)
    
    def tokenize(self, text):
        """
        Tokenize text using the trained BPE model

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        tokens = re.findall(self.pattern, text.lower())
        result = []
        
        # Apply tokenization to each token
        for token in tokens:
            # Check if token is in vocabulary
            if token in self.encoder:
                result.append(self.encoder[token])
                continue
            
            # Split into characters
            chars = list(token)
            
            # Apply merges
            while len(chars) > 1:
                # Find the first valid merge
                merged = False
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    pair_str = f"{pair[0]}+{pair[1]}"
                    if pair_str in self.merges:
                        merged_token = self.merges[pair_str]
                        chars = chars[:i] + [merged_token] + chars[i+2:]
                        merged = True
                        break
                
                if not merged:
                    break
            
            # Convert final pieces to IDs
            for piece in chars:
                if piece in self.encoder:
                    result.append(self.encoder[piece])
                else:
                    # Handle unknown tokens by using the most common unit
                    result.append(self.get_special_token_id('UNK'))
        
        return result
    
    def detokenize(self, token_ids):
        """
        Convert token IDs back to text

        Args:
            token_ids: List of token IDs

        Returns:
            Reconstructed text string
        """
        tokens = [self.decoder.get(token_id, '<UNK>') for token_id in token_ids]
        # Simple join for now - this could be improved with more sophisticated detokenization
        return ''.join(tokens).strip()
    
    def save(self, path):
        """Save tokenizer vocabulary and merges to disk"""
        data = {
            'encoder': self.encoder,
            'special_tokens': self.special_tokens,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved BPE tokenizer to {path}")
    
    @classmethod
    def load(cls, path):
        """Load tokenizer vocabulary and merges from disk"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab_size = data.get('vocab_size', 2000)
        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.encoder = data['encoder']
        # Convert string keys back to int keys for decoder
        tokenizer.decoder = {int(k): v for k, v in data.get('decoder', {}).items()}
        if not tokenizer.decoder:
            # Recreate decoder from encoder if not present
            tokenizer.decoder = {v: k for k, v in tokenizer.encoder.items()}
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.merges = data['merges']
        
        logger.info(f"Loaded BPE tokenizer with {len(tokenizer.encoder)} tokens")
        return tokenizer

def load_or_create_bpe_tokenizer(texts=None, vocab_size=2000, save_path="bpe_tokenizer.json", 
                                special_tokens=None):
    """
    Load BPE tokenizer from file or create and train a new one

    Args:
        texts: List of text strings for training
        vocab_size: Target vocabulary size
        save_path: Path to save/load tokenizer
        special_tokens: Dictionary of special tokens

    Returns:
        Trained BPE tokenizer
    """
    if special_tokens is None:
        special_tokens = {
            'PAD': '<PAD>',
            'SOS': '',  # Start of sequence
            'EOS': '<EOS>',  # End of sequence
            'SEP': '<SEP>',  # Separator
            'UNK': '<UNK>'   # Unknown token
        }
    
    # Try to load existing tokenizer
    if os.path.exists(save_path):
        try:
            logger.info(f"Loading BPE tokenizer from {save_path}")
            tokenizer = BPETokenizer.load(save_path)
            # Verify special tokens exist
            for name in special_tokens:
                if name not in tokenizer.special_tokens:
                    logger.warning(f"Special token '{name}' not found in loaded tokenizer")
                    raise ValueError("Tokenizer missing required special tokens")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Training new tokenizer.")
    
    if not texts:
        raise ValueError("Texts must be provided to train a new tokenizer")
    
    # Create and train new tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.train(texts, save_path=save_path)
    
    return tokenizer
