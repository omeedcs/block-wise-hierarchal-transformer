# PyTorch Transformer Chatbot Implementation

This document provides an extensive overview of a Transformer-based chatbot implemented in PyTorch. It explains what the code does, its importance, how to use it, and how to extend or modify it for further development.

---

## Overview

This codebase implements a Transformer model designed for conversational tasks, such as a chatbot. The model is built from scratch using PyTorch and includes the full pipeline: model architecture, data preparation, training, evaluation, and inference. It supports simple dialogues and can be trained on custom conversation data.

### What the Code Does

- **Model Architecture**: Implements a Transformer decoder with self-attention, embeddings, and feed-forward layers to process and generate text.
- **Data Preparation**: Tokenizes conversation pairs (input and response) and prepares them for training using a custom dataset and DataLoader.
- **Training**: Trains the model using cross-entropy loss and evaluates its performance with perplexity.
- **Inference**: Generates responses to user inputs using top-k sampling.

### Why It’s Important

- **Educational Value**: This implementation serves as a hands-on example of how Transformers work, making it ideal for learning about attention mechanisms, model training, and natural language processing (NLP).
- **Practical Application**: It provides a foundation for building chatbots, which can be extended for real-world applications like customer service or virtual assistants.
- **Flexibility**: The modular design allows for easy experimentation with different configurations, hyperparameters, or datasets.

---

## Code Structure and Components

### 1. Model Architecture

The Transformer model is composed of several modular classes:

#### `Normalization` Class
- **Purpose**: Provides flexible normalization options (layer, batch, or none).
- **Implementation**: Uses `nn.LayerNorm` or `nn.BatchNorm1d` based on the chosen method.
- **Usage**: Stabilizes training by normalizing activations.

#### `FullAttention` Class
- **Purpose**: Computes self-attention scores and applies them to values.
- **Features**: Includes dropout and a fix for compatibility with Apple’s Metal Performance Shaders (MPS).
- **Usage**: Core mechanism allowing the model to focus on relevant parts of the input sequence.

#### `AttentionLayer` Class
- **Purpose**: Projects inputs into queries, keys, and values, then applies the attention mechanism.
- **Features**: Supports multi-head attention with configurable dimensions and dropout.
- **Usage**: Encapsulates attention logic for modularity.

#### `TransformerLayer` Class
- **Purpose**: Defines a single Transformer layer with self-attention and a feed-forward network.
- **Features**: Includes normalization, dropout, and optional activation functions (GELU or ReLU).
- **Usage**: Stacks multiple layers to enable complex pattern learning.

#### `Transformer` Class
- **Purpose**: Combines embeddings, multiple Transformer layers, and an output layer to form the full model.
- **Features**:
  - Token and positional embeddings.
  - Causal attention mask to ensure autoregressive generation.
  - Configurable hyperparameters (e.g., `d_model`, `n_heads`, `layers`).
- **Usage**: The main model for training and inference.

### 2. Data Preparation

- **Vocabulary**: A small, predefined vocabulary with special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<SEP>`).
- **Tokenization**: Converts text into token IDs, adding special tokens to mark sequence boundaries and separate input from response.
- **Dataset**: `ChatDataset` class handles conversation pairs.
- **DataLoader**: Uses `pad_sequence` to batch sequences with padding.

### 3. Training

- **Training Loop**: Iterates over batches, computes cross-entropy loss, and updates model parameters using the Adam optimizer.
- **Evaluation**: Calculates perplexity to measure how well the model predicts the next token.
- **Features**: Includes debug output for monitoring loss per batch and epoch.

### 4. Inference

- **Response Generation**: Uses top-k sampling to generate responses token-by-token.
- **Process**:
  1. Tokenizes the user input with `<SOS>` and `<SEP>`.
  2. Predicts logits and samples the next token from the top-k probabilities.
  3. Continues until `<EOS>` or a maximum length is reached.

---

## How to Use the Code

### Requirements
- Python 3.8+
- PyTorch (`pip install torch`)
- Einops (`pip install einops`) for tensor operations

### Running the Code
1. **Train the Model**:
   - The script trains the model on the provided `raw_conversations` for 200 epochs.
   - Run the script directly: `python transformer_and_train.py`.
   - Model weights are saved to `chat_transformer.pth`.

2. **Evaluate Perplexity**:
   - After training, perplexity is printed to assess model performance.

3. **Test the Chatbot**:
   - Example inputs are provided in `test_inputs`.
   - Responses are generated and printed for each input.

### Example Output
