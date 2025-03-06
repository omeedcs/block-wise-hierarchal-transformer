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
    # Create model
    model = HierarchicalTransformer(
        vocab_size=len(word_to_id),
        d_model=128,
        d_ff=512,
        n_heads=4,
        n_local_layers=2,
        n_global_layers=2,
        block_size=8
    )
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract additional info if available
                epoch = checkpoint.get('epoch', 0)
                loss = checkpoint.get('loss', float('inf'))
                perplexity = checkpoint.get('perplexity', float('inf'))
                print(f"Model loaded from epoch {epoch} with loss {loss:.4f} and perplexity {perplexity:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Model loaded successfully")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Initializing new model instead")
    else:
        print("No checkpoint provided or file not found.")
        print("Initializing new model")
    
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
    try:
        import gradio as gr
        
        # Define conversation response function
        def respond(message, chat_history, temperature=0.8, max_length=20, top_k=10):
            # Input validation
            if not message:
                return chat_history
            
            # Generate response
            temperature = float(temperature)
            max_length = int(max_length)
            top_k = int(top_k)
            
            start_time = time.time()
            response = generate_response(
                message, model, word_to_id, id_to_word,
                max_length=max_length, top_k=top_k, temperature=temperature,
                device=device
            )
            end_time = time.time()
            
            # Log timing information
            print(f"Generated response in {(end_time - start_time):.3f} seconds")
            print(f"User: {message}")
            print(f"Bot: {response}")
            
            # Format for Gradio chatbot (list of [user_msg, bot_response] pairs)
            chat_history.append((message, response))
            return chat_history
        
        # Create interface
        with gr.Blocks(title="Hierarchical Transformer Chatbot") as interface:
            gr.Markdown("# Hierarchical Transformer Chatbot")
            gr.Markdown("This chatbot uses a block-wise hierarchical transformer architecture to generate conversations.")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
                    with gr.Row():
                        submit = gr.Button("Send")
                        clear = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    gr.Markdown("## Parameters")
                    temperature = gr.Slider(0.1, 2.0, 0.8, step=0.1, label="Temperature")
                    max_length = gr.Slider(10, 100, 50, step=1, label="Max Length")
                    top_k = gr.Slider(0, 50, 10, step=1, label="Top K")
            
            # Set up event handlers
            msg.submit(respond, [msg, chatbot, temperature, max_length, top_k], [chatbot], queue=True)
            submit.click(respond, [msg, chatbot, temperature, max_length, top_k], [chatbot], queue=True)
            clear.click(lambda: [], None, chatbot, queue=False)
            
        # Launch interface
        interface.launch(share=True)
        
    except ImportError:
        print("Gradio is not installed. Please install it with: pip install gradio")
        print("Falling back to command-line interface")
        create_cli_interface(model, device)

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
            user_input, model, word_to_id, id_to_word,
            max_length=50, top_k=10, temperature=0.8,
            device=device
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
