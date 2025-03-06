# Hierarchical Transformer Chatbot

This project implements a block-wise hierarchical transformer model for natural language generation and conversation. The architecture processes text in blocks with local and global attention mechanisms for improved efficiency.

## Project Architecture

The hierarchical transformer uses a novel block-wise approach:
1. Text is divided into fixed-size blocks
2. Local transformer layers process each block independently
3. Global transformer layers integrate information across blocks
4. This design balances computational efficiency with modeling long-range dependencies

## Key Features

### Model Architecture
- Block-wise hierarchical transformer
- Configurable model dimensions (d_model, d_ff)
- Adjustable attention heads
- Separate local and global transformer layers
- Customizable block size

### Training System
- Enhanced training with checkpointing
- Learning rate scheduling
- Progress tracking and visualization
- Automatic device selection (CUDA/MPS/CPU)
- Comprehensive metrics collection

### Data Processing
- Multiple data source support (JSON, CSV, TXT)
- Web API integration capabilities
- Synthetic conversation generation
- Template-based conversations
- Robust data cleaning and normalization

### Inference and Evaluation
- Command-line and web interfaces
- Perplexity evaluation
- Response diversity metrics
- Automated conversation testing
- Training curve visualization

### Hardware Support
- CUDA GPU acceleration
- Apple Silicon (MPS) optimization
- CPU fallback with automatic detection

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Gradio (for web interface)
- Additional requirements in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-transformer-chatbot.git
cd hierarchical-transformer-chatbot

# Install dependencies
pip install -r requirements.txt
```

### Training

Train the model using the enhanced training script:

```bash
python3 enhanced_training.py --templates_count 5000 --epochs 20 --batch_size 32
```

Key parameters:
- `--templates_count`: Number of synthetic conversations to generate
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--d_model`: Dimension of the model
- `--n_heads`: Number of attention heads
- See `enhanced_training.py` for all options

### Evaluation

Evaluate model performance with various metrics:

```bash
python3 evaluate_model.py --model_path checkpoints/best_model.pt --eval_mode all
```

Evaluation modes:
- `perplexity`: Calculate perplexity on test data
- `diversity`: Measure response diversity
- `interactive`: Chat interactively with the model
- `auto_convo`: Run automated conversations
- `all`: Run all evaluations

### Visualization

Visualize training results and model performance:

```bash
python3 visualize_results.py --checkpoint_dir checkpoints --output_dir visualizations
```

This generates:
- Training loss and perplexity curves
- Attention pattern visualizations
- Token distribution analysis
- Model comparison metrics

### Chat Interface

Run the chatbot with an interactive interface:

```bash
# Web interface (requires Gradio)
python3 chat_app.py --checkpoint checkpoints/best_model.pt --interface web

# Command-line interface
python3 chat_app.py --checkpoint checkpoints/best_model.pt --interface cli
```

## Usage

### Training the Model

The recommended way to train the model is using the enhanced training script:

```bash
python3 enhanced_training.py --epochs 50 --batch_size 32 --templates 1000 --learning_rate 0.001
```

Key parameters:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--templates`: Number of synthetic conversation templates to generate
- `--learning_rate`: Learning rate for the optimizer
- `--regenerate_data`: Force regeneration of training data

The hierarchical transformer script will automatically forward to the enhanced training script when run directly.

### Using the Chat Interface

To interact with the trained model:

```bash
python3 chat_app.py
```

This will start a Gradio web interface that allows you to chat with the model.

## Project Structure

- `hiearchal_transformer.py`: Core model implementation
- `data_loader.py`: Data loading and processing utilities
- `enhanced_training.py`: Advanced training script with metrics tracking
- `evaluate_model.py`: Model evaluation tools
- `visualize_results.py`: Visualization utilities for training results
- `chat_app.py`: Interactive chat interfaces (CLI and web)
- `test_chat.py`: Testing script for the chat server
- `checkpoints/`: Directory for model checkpoints
- `visualizations/`: Directory for generated visualizations

## Acknowledgements

This project draws inspiration from:
- The transformer architecture proposed by Vaswani et al. in "Attention Is All You Need"
- Hierarchical attention mechanisms for efficient NLP processing
- Advances in conversational AI and neural dialogue systems

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

## Hardware Compatibility

The model is designed to run on multiple hardware backends:

### CUDA GPUs
- Full support for NVIDIA GPUs through PyTorch's CUDA backend
- Automatically utilized if available on the system

### Apple Silicon (MPS)
- Support for Apple M-series chips through Metal Performance Shaders
- Includes fallback mechanisms to handle MPS-specific limitations
- Automatically falls back to CPU for operations with compatibility issues

### CPU
- Always available as a fallback option
- Used when specialized hardware is not available or encounters errors

### Automatic Device Selection
The code automatically selects the best available device in this order:
1. CUDA GPU (if available)
2. Apple Silicon MPS (if available, with CPU fallback for problematic operations)
3. CPU (as the universal fallback)

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

## References

1. Vaswani, A., et al. (2017). "Attention is All You Need"
2. Dai, Z., et al. (2019). "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" 
3. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer"
4. Yang, Z., et al. (2016). "Hierarchical Attention Networks for Document Classification"
5. Tay, Y., et al. (2020). "Efficient Transformers: A Survey"
