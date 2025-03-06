# Hierarchical Transformer: A Block-wise Approach to Sequence Modeling

This repository contains an implementation of a Hierarchical Transformer model that processes sequences in a block-wise manner. The model is inspired by research on hierarchical attention mechanisms and long-sequence modeling techniques, combining elements from papers such as "Hierarchical Transformers for Long Document Classification" and "Longformer: The Long-Document Transformer."

## Model Architecture & Innovation

### Hierarchical Processing
The model introduces a two-level hierarchical structure:
1. **Local Processing**: First-level processing within fixed-size blocks (default: 8 tokens)
2. **Global Processing**: Second-level processing that operates on block-level representations

This approach reduces the quadratic attention complexity inherent in standard transformers and enables more efficient processing of longer sequences.

### Key Components

#### 1. `HierarchicalTransformer` Class
The main model class that orchestrates:
- Token embeddings and positional encoding
- Local processing of token blocks
- Pooling of block representations
- Global processing across blocks
- Final output projection

#### 2. `FullAttention` Class
- Implements scaled dot-product attention with numerical stability enhancements
- Uses optimized softmax computation in log-space
- Includes gradient clipping for training stability
- Compatible with various hardware backends including Apple MPS

#### 3. `AttentionLayer` Class 
- Encapsulates multi-head attention
- Projects inputs into queries, keys, and values
- Supports different dimensions for queries, keys, and values
- Includes layer normalization for each projection

#### 4. `TransformerLayer` Class
- Combines attention and feed-forward networks
- Uses residual connections and layer normalization
- Configurable number of attention heads and feed-forward dimensions

#### 5. `PoolingLayer` Class
- Specialized attention-based pooling for block summarization
- Uses a learnable query parameter to extract relevant information
- Crucial for the transition between local and global processing levels

#### 6. `FeedForward` Class
- Implements a position-wise feed-forward network with GELU activation
- Uses two linear transformations with a non-linearity in between
- Includes dropout and layer normalization

## Technical Details

### Local Processing
Sequences are first divided into fixed-size blocks of `block_size` tokens. Each block is processed independently by `n_local_layers` transformer layers, allowing parallel computation. This captures local patterns and dependencies within each block.

### Block Aggregation
After local processing, the `PoolingLayer` uses attention-based pooling to create a single representation for each block. This reduces the sequence length by a factor of `block_size` while preserving the most important information from each block.

### Global Processing
The pooled block representations are then processed by `n_global_layers` transformer layers, allowing the model to capture dependencies between different blocks. This enables long-range interactions without the quadratic complexity of standard transformers.

### Training Stability Enhancements
Several techniques are employed to ensure training stability:
- Gradient clipping throughout the forward pass
- Improved numerical stability in attention calculations
- Careful parameter initialization
- Early detection of NaN/Inf values during training

## Relation to Research

### Research Foundations
This implementation draws inspiration from several research directions:

1. **Hierarchical Transformers**:
   - Similar to the approach in "Hierarchical Transformers for Document Understanding" where documents are processed at multiple granularity levels.
   
2. **Efficient Attention Mechanisms**:
   - Takes inspiration from "Efficient Transformers: A Survey" by incorporating a hierarchical structure to reduce complexity.
   
3. **Long-range Sequence Modeling**:
   - Addresses challenges discussed in "Long Range Arena: A Benchmark for Efficient Transformers" by using a block-wise approach.

### Advantages Over Standard Transformers
- **Computational Efficiency**: Reduces the O(n²) attention complexity by operating on blocks
- **Memory Efficiency**: Requires less memory for processing long sequences
- **Scalability**: Better suited for longer sequences than standard transformers
- **Hierarchical Understanding**: Natural for data with inherent hierarchical structure (like conversations or documents)

## Implementation Details

### Data Preprocessing
- Conversations are tokenized and formatted with special tokens (`SOS`, `EOS`, `SEP`)
- Sequences are padded to ensure divisibility by block size
- Masks are created to handle padded tokens appropriately

### Training Process
- Uses cross-entropy loss with padding token ignored
- Employs the Adam optimizer with learning rate 5e-4
- Includes gradient clipping to prevent exploding gradients
- Trains for multiple epochs with batch size 8

### Inference
- Uses top-k sampling with adjustable temperature
- Generates responses token-by-token in an autoregressive manner
- Handles block-wise processing during generation

## Usage Guide

### Requirements
- Python 3.6+
- PyTorch 1.7+
- Math, Torch, and other standard libraries

### Training
```bash
python hiearchal_transformer.py
```
The script will train the model on a small dataset of conversations and evaluate its performance.

### Hyperparameters
Key hyperparameters include:
- `block_size`: Size of local processing blocks (default: 8)
- `d_model`: Model dimension (default: 128)
- `n_heads`: Number of attention heads (default: 4)
- `n_local_layers`: Number of local transformer layers (default: 2)
- `n_global_layers`: Number of global transformer layers (default: 2)

## Potential Applications

This hierarchical transformer can be adapted for various NLP tasks:
- **Document Classification**: Processing long documents by capturing both local and global context
- **Text Summarization**: Extracting key information at different granularity levels
- **Long-form Question Answering**: Understanding questions and generating coherent long answers
- **Dialogue Systems**: Modeling conversation history efficiently

## Future Directions

Potential enhancements to explore:
1. **Sparse Attention**: Incorporating sparse attention patterns in the global layers
2. **Dynamic Block Sizing**: Adaptive block sizes based on content boundaries
3. **Cross-modal Hierarchies**: Extending to multi-modal inputs like text and images
4. **Pre-training Objectives**: Developing specialized pre-training tasks for hierarchical models

## Limitations

Current limitations to be aware of:
- Requires careful tuning of hyperparameters for optimal performance
- Block boundaries may not align with natural language segments
- May lose fine-grained token relationships during pooling
- Currently implemented for a small-scale demonstration

## Acknowledgments

This implementation draws inspiration from:
- The original Transformer paper (Vaswani et al., 2017)
- Research on hierarchical attention networks
- The PyTorch library and community

## References

1. Vaswani, A., et al. (2017). "Attention is All You Need"
2. Dai, Z., et al. (2019). "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" 
3. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer"
4. Yang, Z., et al. (2016). "Hierarchical Attention Networks for Document Classification"
5. Tay, Y., et al. (2020). "Efficient Transformers: A Survey"
