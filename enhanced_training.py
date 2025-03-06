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
    calculate_perplexity
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
    checkpoint_interval=1
):
    """
    Enhanced training function with checkpointing and metrics tracking
    
    Args:
        model: The hierarchical transformer model to train
        dataset: The dataset to train on
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on (cuda, mps, cpu)
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Epoch interval for saving checkpoints
    
    Returns:
        Tuple of (losses, perplexities) for tracking training progress
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
    
    model = model.to(device)
    logger.info(f"Training on device: {device}")
    
    # Create optimizer with gradient clipping
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track metrics
    epoch_losses = []
    perplexities = []
    best_perplexity = float('inf')
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (tokens, pad_mask) in enumerate(progress_bar):
            tokens = tokens.to(device)
            pad_mask = pad_mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(tokens, pad_mask)
            
            # Ensure shapes match for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            targets = tokens[:, :seq_len].contiguous()
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / batch_count
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / batch_count
        epoch_losses.append(epoch_loss)
        
        # Calculate perplexity
        model.eval()
        perplexity = calculate_perplexity(model, dataloader, PAD_TOKEN_ID, device)
        perplexities.append(perplexity)
        model.train()
        
        # Update learning rate based on perplexity
        scheduler.step(perplexity)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Save checkpoint if specified
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'perplexity': perplexity
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'perplexity': perplexity
            }, best_model_path)
            logger.info(f"New best model saved with perplexity: {perplexity:.4f}")
    
    # Plot training progress
    plot_training_progress(epoch_losses, perplexities)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocabulary': {
            'word_to_id': word_to_id,
            'id_to_word': id_to_word
        },
        'params': {
            'vocab_size': len(word_to_id),
            'd_model': model.d_model,
            'block_size': model.block_size
        }
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return epoch_losses, perplexities

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
    if os.path.exists(output_file) and not force_regenerate:
        logger.info(f"Loading existing conversation data from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Generate new data
    logger.info(f"Generating {templates_count} synthetic conversations")
    processor = DataProcessor()
    
    # Add synthetic data source
    templates = get_conversation_templates()
    synthetic = SyntheticDataSource(templates, count=templates_count)
    processor.add_source(synthetic)
    
    # Process all data
    conversations = processor.process_all_sources()
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Saved {len(conversations)} conversations to {output_file}")
    return conversations

def main(args):
    # Load or create enhanced dataset
    conversations = load_or_create_synthetic_data(
        templates_count=args.templates_count,
        output_file=args.data_file,
        force_regenerate=args.regenerate_data
    )
    
    # Tokenize conversations
    logger.info("Tokenizing conversations...")
    tokenized_conversations = []
    for conv in conversations:
        tokenized_conversations.extend(
            tokenize_conversation(conv, word_to_id, SOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID)
        )
    
    # Create dataset
    dataset = ChatDataset(tokenized_conversations, max_seq_len=args.max_seq_len, pad_token_id=PAD_TOKEN_ID)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Initialize model
    model = HierarchicalTransformer(
        vocab_size=len(word_to_id),
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_local_layers=args.n_local_layers,
        n_global_layers=args.n_global_layers,
        block_size=args.block_size
    )
    
    # Determine device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Train model
    logger.info("Starting enhanced training...")
    train_model_enhanced(
        model=model,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    logger.info("Training complete!")

# Enhanced version of calculate_perplexity that accepts a device parameter
def calculate_perplexity(model, dataloader, pad_token_id, device=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Use the provided device or the model's current device
    if device is None:
        device = next(model.parameters()).device
    
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
            
    # Handle edge case to avoid division by zero
    if total_tokens == 0:
        return float('inf')
        
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced training for Hierarchical Transformer")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="enhanced_conversations.json",
                        help="File to save/load conversation data")
    parser.add_argument("--templates_count", type=int, default=5000,
                        help="Number of synthetic conversations to generate")
    parser.add_argument("--regenerate_data", action="store_true",
                        help="Force regeneration of data even if file exists")
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="Maximum sequence length for training")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=512,
                        help="Feed-forward dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_local_layers", type=int, default=2,
                        help="Number of local transformer layers")
    parser.add_argument("--n_global_layers", type=int, default=2,
                        help="Number of global transformer layers")
    parser.add_argument("--block_size", type=int, default=8,
                        help="Block size for hierarchical processing")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Epoch interval for saving checkpoints")
    
    args = parser.parse_args()
    main(args)
