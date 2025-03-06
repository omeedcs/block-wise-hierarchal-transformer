"""
Visualization tools for analyzing training and evaluation results.

This script provides functions to visualize:
1. Training loss and perplexity
2. Attention patterns in the transformer model
3. Token distributions
4. Response diversity measurements
5. Model comparison metrics

Usage:
    python3 visualize_results.py --checkpoint_dir checkpoints --output_dir visualizations
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import glob
import pandas as pd
import logging

# Import from our modules
from hiearchal_transformer import (
    HierarchicalTransformer, tokenize_conversation, generate_response,
    SOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, 
    vocab, word_to_id, id_to_word
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Visualization")

def load_checkpoints(checkpoint_dir):
    """
    Load training metrics from checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Dictionary with training metrics
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    metrics = {
        'epoch': [],
        'loss': [],
        'perplexity': []
    }
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            metrics['epoch'].append(checkpoint.get('epoch', 0))
            metrics['loss'].append(checkpoint.get('loss', float('inf')))
            metrics['perplexity'].append(checkpoint.get('perplexity', float('inf')))
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
    
    return metrics

def plot_training_curves(metrics, output_dir):
    """
    Plot training loss and perplexity curves.
    
    Args:
        metrics: Dictionary with training metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['loss'], 'b-', marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    
    # Plot perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['perplexity'], 'r-', marker='o')
    plt.title('Model Perplexity over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_perplexity.png'))

def load_model(model_path, device=None):
    """
    Load a model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on (cuda, mps, cpu)
        
    Returns:
        The loaded model
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    logger.info(f"Loading model from {model_path} to {device}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with same parameters
        model = HierarchicalTransformer(
            vocab_size=len(word_to_id),
            d_model=128,  # Could be stored in checkpoint
            d_ff=512,     # Could be stored in checkpoint
            n_heads=4,    # Could be stored in checkpoint
            n_local_layers=2,  # Could be stored in checkpoint
            n_global_layers=2  # Could be stored in checkpoint
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def visualize_attention(model, input_text, output_dir, device=None):
    """
    Visualize attention patterns for a given input.
    
    Args:
        model: Trained model
        input_text: Text input to analyze
        output_dir: Directory to save visualizations
        device: Device to run model on
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize input
    tokens = [SOS_TOKEN_ID] + [word_to_id.get(word, word_to_id["<unk>"]) for word in input_text.split()]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Create attention mask
    pad_mask = torch.zeros(1, input_tensor.size(1), 1, dtype=torch.bool).to(device)
    
    # Run model with hooks to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Extract attention weights from output
        attn_output, attn_weights = output
        attention_weights.append(attn_weights)
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if "attn_layer" in name and hasattr(module, "forward"):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor, pad_mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize attention maps
    os.makedirs(output_dir, exist_ok=True)
    
    input_tokens = [id_to_word[tid.item()] for tid in input_tensor[0]]
    
    for i, attn_map in enumerate(attention_weights):
        # Get attention weights
        attn = attn_map[0].cpu().numpy()  # Shape: [head, seq_len, seq_len]
        
        # Plot each attention head
        for h in range(attn.shape[0]):
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attn[h], cmap='viridis', 
                xticklabels=input_tokens, 
                yticklabels=input_tokens
            )
            plt.title(f'Attention layer {i}, head {h}')
            plt.xlabel('Key tokens')
            plt.ylabel('Query tokens')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'attention_layer{i}_head{h}.png'))
            plt.close()

def analyze_token_distributions(model, test_inputs, output_dir, device=None, num_responses=50):
    """
    Analyze token distributions in model responses.
    
    Args:
        model: Trained model
        test_inputs: List of test inputs
        output_dir: Directory to save results
        device: Device to run model on
        num_responses: Number of responses to generate per input
    """
    if device is None:
        device = next(model.parameters()).device
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate responses
    all_tokens = []
    response_lengths = []
    
    for input_text in tqdm(test_inputs, desc="Generating responses"):
        for _ in range(num_responses):
            response = generate_response(
                model, input_text, word_to_id, id_to_word,
                SOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
                device=device, temperature=1.0
            )
            
            tokens = response.split()
            all_tokens.extend(tokens)
            response_lengths.append(len(tokens))
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Plot token frequency distribution (top 30)
    top_tokens = token_counts.most_common(30)
    token_df = pd.DataFrame(top_tokens, columns=['Token', 'Count'])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Count', y='Token', data=token_df)
    plt.title('Top 30 Tokens in Generated Responses')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_frequency.png'))
    
    # Plot response length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(response_lengths, bins=20, kde=True)
    plt.title('Response Length Distribution')
    plt.xlabel('Response Length (tokens)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_length.png'))
    
    # Save raw token counts
    with open(os.path.join(output_dir, 'token_counts.json'), 'w') as f:
        # Convert counter to dict and sort by frequency
        token_dict = {k: v for k, v in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)}
        json.dump(token_dict, f, indent=2)

def compare_models(model_paths, test_inputs, output_dir, device=None):
    """
    Compare performance metrics across different model checkpoints.
    
    Args:
        model_paths: List of paths to model checkpoints
        test_inputs: List of test inputs
        output_dir: Directory to save results
        device: Device to run models on
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to track
    metrics = {
        'model': [],
        'avg_response_time': [],
        'avg_response_length': [],
        'unique_token_ratio': []
    }
    
    # Compare models
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        logger.info(f"Evaluating model: {model_name}")
        
        # Load model
        model = load_model(model_path, device)
        
        # Track metrics
        response_times = []
        response_lengths = []
        all_tokens = []
        
        # Generate responses
        for input_text in tqdm(test_inputs, desc=f"Testing {model_name}"):
            # Time response generation
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            response = generate_response(
                model, input_text, word_to_id, id_to_word,
                SOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
                device=device, temperature=0.8
            )
            end_time.record()
            
            # Wait for CUDA kernels to finish
            torch.cuda.synchronize()
            
            # Track metrics
            response_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            response_times.append(response_time)
            
            tokens = response.split()
            response_lengths.append(len(tokens))
            all_tokens.extend(tokens)
        
        # Calculate metrics
        metrics['model'].append(model_name)
        metrics['avg_response_time'].append(np.mean(response_times))
        metrics['avg_response_length'].append(np.mean(response_lengths))
        metrics['unique_token_ratio'].append(len(set(all_tokens)) / len(all_tokens) if all_tokens else 0)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Plot comparison charts
    for metric in ['avg_response_time', 'avg_response_length', 'unique_token_ratio']:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, data=metrics_df)
        plt.title(f'Model Comparison: {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'compare_{metric}.png'))
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

def main(args):
    """Main function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize training curves
    if args.visualize_training:
        logger.info("Visualizing training curves...")
        metrics = load_checkpoints(args.checkpoint_dir)
        plot_training_curves(metrics, os.path.join(args.output_dir, 'training'))
    
    # Prepare test inputs
    test_inputs = [
        "Hello, how are you?",
        "What do you think about artificial intelligence?",
        "Tell me a joke",
        "What's the weather like today?",
        "Can you help me with my homework?",
        "What's your favorite movie?",
        "How do I learn programming?",
        "Tell me about yourself",
        "What's the meaning of life?",
        "What can you do?"
    ]
    
    # Analyze best model
    if args.visualize_attention or args.analyze_tokens:
        logger.info("Loading best model...")
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model = load_model(best_model_path)
            
            if args.visualize_attention:
                logger.info("Visualizing attention patterns...")
                for input_text in test_inputs[:3]:  # Only visualize first 3 inputs
                    visualize_attention(
                        model, 
                        input_text, 
                        os.path.join(args.output_dir, 'attention', input_text.replace(' ', '_')[:20])
                    )
            
            if args.analyze_tokens:
                logger.info("Analyzing token distributions...")
                analyze_token_distributions(
                    model,
                    test_inputs,
                    os.path.join(args.output_dir, 'token_analysis')
                )
        else:
            logger.warning(f"Best model not found at {best_model_path}")
    
    # Compare models
    if args.compare_models:
        logger.info("Comparing models...")
        # Find all checkpoint files
        model_paths = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch_*.pt"))
        # Add best model if exists
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model_paths.append(best_model_path)
        
        # Sort by epoch
        model_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else float('inf'))
        
        # Only keep models at specified intervals to avoid too many comparisons
        if len(model_paths) > args.max_models_to_compare:
            interval = len(model_paths) // args.max_models_to_compare
            model_paths = model_paths[::interval]
            # Always include the last model
            if model_paths[-1] != model_paths[-1]:
                model_paths.append(model_paths[-1])
        
        if model_paths:
            compare_models(
                model_paths,
                test_inputs,
                os.path.join(args.output_dir, 'model_comparison')
            )
        else:
            logger.warning("No model checkpoints found for comparison")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training and evaluation results")
    
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument('--output_dir', type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument('--visualize_training', action='store_true', default=True,
                        help="Visualize training metrics")
    parser.add_argument('--visualize_attention', action='store_true', default=False,
                        help="Visualize attention patterns")
    parser.add_argument('--analyze_tokens', action='store_true', default=True,
                        help="Analyze token distributions in responses")
    parser.add_argument('--compare_models', action='store_true', default=False,
                        help="Compare performance across model checkpoints")
    parser.add_argument('--max_models_to_compare', type=int, default=5,
                        help="Maximum number of models to include in comparison")
    
    args = parser.parse_args()
    main(args)
