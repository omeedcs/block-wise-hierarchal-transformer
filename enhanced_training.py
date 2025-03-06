"""
Enhanced Training Script for Hierarchical Transformer Model

This script uses the data_loader module to fetch conversation data from various 
sources and trains the hierarchical transformer model with this enhanced dataset.
"""

import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math

# Import the data loader
from data_loader import (
    DataProcessor, 
    SyntheticDataSource, 
    JSONDataSource, 
    CSVDataSource, 
    TXTDataSource,
    WebAPIDataSource,
    get_conversation_templates
)

# Import model components
from hiearchal_transformer import (
    HierarchicalTransformer,
    tokenize_conversation,
    word_to_id,
    id_to_word,
    SOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SEP_TOKEN_ID,
    PAD_TOKEN_ID,
    ChatDataset,
    collate_fn,
    VOCAB_SIZE,
    load_or_create_vocab,
    tokenize
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EnhancedTraining")

def plot_training_progress(losses, perplexities, save_path="training_progress.png"):
    """Plot and save training progress"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    epochs = range(1, len(losses) + 1)
    ax1.plot(epochs, losses, 'b-')
    ax1.set_title('Training Loss by Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot perplexity
    ax2.plot(epochs, perplexities, 'r-')
    ax2.set_title('Perplexity by Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training progress plot saved to {save_path}")

def train_model_enhanced(
    model, 
    dataset, 
    num_epochs=10, 
    batch_size=8, 
    learning_rate=0.001,
    device=None,
    checkpoint_dir="checkpoints",
    checkpoint_interval=1,
    warmup_steps=500
):
    """
    Enhanced training function with checkpointing, metrics tracking, and learning rate warmup
    
    Args:
        model: The hierarchical transformer model to train
        dataset: The dataset to train on
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on (cuda, mps, cpu)
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Epoch interval for saving checkpoints
        warmup_steps: Number of warmup steps for learning rate scheduler
    
    Returns:
        Tuple of (losses, perplexities) for tracking training progress
    """
    logger = logging.getLogger("EnhancedTraining")
    
    # Determine device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else 
                            "cpu")
    
    # Move model to the appropriate device
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    # Create dataloader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=dataset.pad_token_id)
    )
    
    # Use AdamW optimizer for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Create learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track best model perplexity
    best_perplexity = float('inf')
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # Lists to track metrics
    losses = []
    perplexities = []
    total_steps = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (input_ids, target_ids, pad_mask) in enumerate(progress_bar):
            total_steps += 1
            
            # Move tensors to the correct device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            pad_mask = pad_mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, pad_mask)
            
            # Reshape for cross-entropy loss
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            # Calculate loss (ignore padding tokens)
            loss = F.cross_entropy(
                logits, 
                target_ids, 
                ignore_index=dataset.pad_token_id
            )
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1), lr=f"{current_lr:.6f}")
            
            # Track batch loss
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Evaluate perplexity
        model.eval()
        perplexity = calculate_perplexity(model, dataloader, dataset.pad_token_id, device=device)
        perplexities.append(perplexity)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'perplexity': perplexity
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'perplexity': perplexity
            }, best_model_path)
            logger.info(f"New best model saved with perplexity: {perplexity:.4f}")
    
    return losses, perplexities

def load_or_create_synthetic_data(
    templates_count=1000, 
    output_file="enhanced_conversations.json",
    force_regenerate=False
):
    """
    Load existing synthetic data or create new data if needed
    
    Args:
        templates_count: Number of synthetic conversations to generate
        output_file: File to save/load data from
        force_regenerate: Whether to force regeneration even if file exists
        
    Returns:
        List of conversation turns
    """
    logger = logging.getLogger("EnhancedTraining")
    
    # Check if data already exists and we're not forcing regeneration
    if os.path.exists(output_file) and not force_regenerate:
        logger.info(f"Loading existing data from {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Generating {templates_count} synthetic conversations")
    
    # Setup data sources and processor
    data_processor = DataProcessor()
    
    # Add synthetic data source
    templates = get_conversation_templates()
    synthetic_source = SyntheticDataSource(templates, count=templates_count)
    data_processor.add_source(synthetic_source)
    
    # Process data
    conversations = data_processor.process_all_sources()
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Generated and saved {len(conversations)} conversations to {output_file}")
    return conversations

def main(args):
    # Set up logging
    logger = logging.getLogger("EnhancedTraining")
    
    # Load or create enhanced dataset
    conversations = load_or_create_synthetic_data(
        templates_count=args.templates,
        output_file=args.output_file,
        force_regenerate=args.force_regen
    )
    
    # Import model with updated vocab size
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Re-import with updated vocab parameters (this will trigger vocab loading)
    from hiearchal_transformer import (
        HierarchicalTransformer,
        tokenize_conversation,
        word_to_id,
        id_to_word,
        SOS_TOKEN_ID,
        EOS_TOKEN_ID,
        SEP_TOKEN_ID,
        PAD_TOKEN_ID,
        ChatDataset,
        collate_fn,
        VOCAB_SIZE,
        load_or_create_vocab,
        tokenize
    )
    
    # Ensure vocabulary is large enough
    if args.vocab_size > VOCAB_SIZE:
        logger.info(f"Expanding vocabulary from {VOCAB_SIZE} to {args.vocab_size} words")
        load_or_create_vocab(min_vocab_size=args.vocab_size)
    else:
        logger.info(f"Using existing vocabulary with {VOCAB_SIZE} words")
    
    # Process conversations - properly tokenize them
    tokenized_conversations = []
    logger.info(f"Tokenizing {len(conversations)} conversations...")
    
    for conversation in conversations:
        # Each conversation is a list of strings (turns)
        tokenized_turns = []
        
        # Process each turn in the conversation
        for i in range(len(conversation) - 1):  # Minus 1 because we need pairs
            # Format: [SOS, input_tokens, SEP, response_tokens, EOS]
            input_turn = conversation[i]
            response_turn = conversation[i + 1]
            
            # Tokenize the turns
            input_tokens = tokenize(input_turn)
            response_tokens = tokenize(response_turn)
            
            # Combine with special tokens
            combined = [SOS_TOKEN_ID] + input_tokens + [SEP_TOKEN_ID] + response_tokens + [EOS_TOKEN_ID]
            tokenized_turns.append(combined)
        
        tokenized_conversations.extend(tokenized_turns)
    
    logger.info(f"Created {len(tokenized_conversations)} training samples")
    
    # Create dataset
    dataset = ChatDataset(tokenized_conversations, max_seq_len=64, pad_token_id=PAD_TOKEN_ID)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Initialize model
    model = HierarchicalTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_local_layers=args.n_local_layers,
        n_global_layers=args.n_global_layers,
        block_size=args.block_size
    )
    
    # Determine device
    device = None
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Check if we should load existing model
    model_path = "hierarchical_transformer_enhanced.pth"
    try:
        if os.path.exists(model_path) and not args.force_new:
            logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model: {e}")
        logger.info("Initializing new model instead")
    
    # Move model to device
    model = model.to(device)
    
    # Train model
    logger.info("Starting enhanced training...")
    train_model_enhanced(
        model=model,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        warmup_steps=args.warmup_steps
    )
    
    logger.info("Training complete!")

# Enhanced version of calculate_perplexity that accepts a device parameter
def calculate_perplexity(model, dataloader, pad_token_id, device=None):
    """
    Calculate perplexity on the dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with validation data
        pad_token_id: ID of padding token to ignore in loss calculation
        device: Device to run evaluation on
        
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    
    # Determine device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for input_tensor, target_tensor, pad_mask in dataloader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            pad_mask = pad_mask.to(device)
            
            # Forward pass
            logits = model(input_tensor, pad_mask)
            
            # Calculate loss (ignore padding tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tensor.view(-1),
                ignore_index=pad_token_id,
                reduction='sum'
            )
            
            # Count non-padding tokens
            non_pad_mask = (target_tensor != pad_token_id)
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # Calculate perplexity (exp of average negative log-likelihood)
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced training for Hierarchical Transformer")
    
    # Data parameters
    parser.add_argument("--templates", type=int, default=1000, help="Number of synthetic conversation templates to generate")
    parser.add_argument("--force_regen", action="store_true", help="Force regeneration of data even if it exists")
    parser.add_argument("--output_file", type=str, default="enhanced_conversations.json", help="Output file for synthetic data")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of model")
    parser.add_argument("--d_ff", type=int, default=512, help="Dimension of feedforward network")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_local_layers", type=int, default=2, help="Number of local transformer layers")
    parser.add_argument("--n_global_layers", type=int, default=2, help="Number of global transformer layers")
    parser.add_argument("--block_size", type=int, default=8, help="Size of input blocks")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Minimum vocabulary size to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval to save checkpoints (epochs)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], help="Device to train on (default: auto-detect)")
    parser.add_argument("--force_new", action="store_true", help="Force training a new model instead of loading existing one")
    
    args = parser.parse_args()
    
    # Pass the vocabulary size to the main function
    main(args)
