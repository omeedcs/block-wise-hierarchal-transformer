import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template_string, request, jsonify
import argparse
import time
import gradio as gr

# Check if hiearchal_transformer.py exists in the current directory
if not os.path.exists('hiearchal_transformer.py'):
    print("Error: hiearchal_transformer.py not found in the current directory.")
    sys.exit(1)

# Import the required components from hiearchal_transformer.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hiearchal_transformer import (
    HierarchicalTransformer, 
    FullAttention, 
    word_to_id, 
    id_to_word, 
    SOS_TOKEN_ID, 
    EOS_TOKEN_ID, 
    SEP_TOKEN_ID, 
    PAD_TOKEN_ID,
    generate_response
)

# Flask application setup
app = Flask(__name__)

# Load the trained model
def load_model(checkpoint_path=None):
    """
    Load the model from a checkpoint if provided, otherwise initialize a new model.
    
    Args:
        checkpoint_path: Path to a model checkpoint file
        
    Returns:
        Loaded or initialized model
    """
    # Create model - use parameters matching our training
    model = HierarchicalTransformer(
        vocab_size=2000,  # Match the checkpoint's vocabulary size
        d_model=256,
        d_ff=1024,
        n_heads=8,
        block_size=8,
        n_local_layers=2,  # Match the checkpoint's layer count
        n_global_layers=2
    )
    
    # Check if a checkpoint path is provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Check if the checkpoint contains a model_state_dict key (from enhanced training)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {checkpoint_path} (epoch: {checkpoint.get('epoch', 'unknown')})")
                print(f"Perplexity: {checkpoint.get('perplexity', 'unknown')}")
            else:
                # Direct state dict format
                model.load_state_dict(checkpoint)
                print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load model from {checkpoint_path}: {e}")
            print("Initializing new model instead")
    else:
        # Look for default model location
        default_path = "hierarchical_transformer_enhanced.pth"
        if os.path.exists(default_path):
            try:
                # Determine device
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
                    
                # Load the checkpoint
                checkpoint = torch.load(default_path, map_location=device)
                # Check if the checkpoint contains a model_state_dict key (from enhanced training)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Model loaded from {default_path} (epoch: {checkpoint.get('epoch', 'unknown')})")
                    print(f"Perplexity: {checkpoint.get('perplexity', 'unknown')}")
                else:
                    # Direct state dict format
                    model.load_state_dict(checkpoint)
                    print(f"Model loaded from {default_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Initializing new model instead")
        else:
            print("No model checkpoint found. Initializing new model.")
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

# Create a simple web interface using Gradio
def create_web_interface(model, device):
    """
    Create a web interface for the chatbot using Gradio.
    
    Args:
        model: The transformer model to use
        device: Device to run inference on
    """
    conversation_history = []
    
    def respond(message, history):
        # Add the user message to our conversation history
        conversation_history.append(f"User: {message}")
        
        # Start time for response generation
        start_time = time.time()
        
        # Generate response using our model
        try:
            response = generate_response(
                model, 
                message, 
                word_to_id, 
                device, 
                max_length=40, 
                temperature=0.7, 
                top_k=30,
                repetition_penalty=1.5
            )
            
            # Calculate the time taken to generate response
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"Generated response in {generation_time:.3f} seconds")
            
            # Add the response to our conversation history
            conversation_history.append(f"Bot: {response}")
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing that. Could you try again?"
    
    # Create the Gradio interface
    gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=500),
        title="Hierarchical Transformer Chatbot",
        description="A block-wise hierarchical transformer chatbot",
        theme="default",
        examples=[
            "Hello, how are you?",
            "What's your name?",
            "Tell me about yourself",
            "What can you do?",
            "Thank you for your help"
        ],
        cache_examples=False
    ).launch(share=True)

def create_cli_interface(model, device):
    """
    Create a command-line interface for the chatbot.
    
    Args:
        model: The transformer model to use
        device: Device to run inference on
    """
    print("\n" + "="*50)
    print("Hierarchical Transformer Chatbot")
    print("Type 'exit' to end the conversation.")
    print("="*50 + "\n")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        conversation_history.append(user_input)
        
        # Generate response
        start_time = time.time()
        response = generate_response(
            model, 
            user_input, 
            word_to_id, 
            device,
            max_length=40, 
            temperature=0.7, 
            top_k=30,
            repetition_penalty=1.5
        )
        end_time = time.time()
        
        conversation_history.append(response)
        
        # Print response with timing information
        print(f"Bot: {response}")
        print(f"(Response generated in {(end_time - start_time):.3f} seconds)")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Transformer Chatbot")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument('--interface', type=str, choices=['web', 'cli'], default='web',
                        help="Interface type (web or command-line)")
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.checkpoint)
    
    # Create interface
    if args.interface == 'web':
        try:
            create_web_interface(model, device)
        except Exception as e:
            print(f"Error creating web interface: {e}")
            print("Falling back to command-line interface")
            create_cli_interface(model, device)
    else:
        create_cli_interface(model, device)
