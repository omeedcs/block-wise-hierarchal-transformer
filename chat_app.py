import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template_string, request, jsonify

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
def load_model():
    # Model hyperparameters
    VOCAB_SIZE = len(word_to_id)
    D_MODEL = 128
    D_FF = 512
    N_HEADS = 4
    N_LOCAL_LAYERS = 2
    N_GLOBAL_LAYERS = 2
    BLOCK_SIZE = 8
    
    # Initialize the model
    model = HierarchicalTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_ff=D_FF,
        n_heads=N_HEADS,
        n_local_layers=N_LOCAL_LAYERS,
        n_global_layers=N_GLOBAL_LAYERS,
        block_size=BLOCK_SIZE
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        try:
            model = model.cuda()
        except:
            device = torch.device("cpu")
            print("Failed to use CUDA. Using CPU instead.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            model = model.to(device)
        except:
            device = torch.device("cpu")
            print("Failed to use MPS. Using CPU instead.")
    else:
        print("Using CPU for inference.")
    
    print(f"Model loaded on {device}")
    return model, device

# Generate a response using the model
def get_model_response(user_input, model, device, max_length=20, top_k=5, temperature=0.7):
    try:
        # Try first with the preferred device
        response = generate_response(
            user_input, 
            model, 
            word_to_id, 
            id_to_word, 
            max_length=max_length, 
            top_k=top_k, 
            temperature=temperature,
            device=device
        )
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        try:
            # Fall back to CPU if there's an issue
            print("Falling back to CPU for generation")
            cpu_device = torch.device("cpu")
            model_cpu = model.to(cpu_device)
            response = generate_response(
                user_input, 
                model_cpu, 
                word_to_id, 
                id_to_word, 
                max_length=max_length, 
                top_k=top_k, 
                temperature=temperature,
                device=cpu_device
            )
            # Move model back to original device after generation
            model.to(device)
            return response
        except Exception as e2:
            print(f"Error generating response on CPU: {e2}")
            return "Sorry, I had trouble processing that. Can you try again?"

# HTML template with integrated CSS and JavaScript
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hierarchical Transformer Chat</title>
    <style>
        :root {
            --primary-color: #6a11cb;
            --secondary-color: #2575fc;
            --text-color: #333;
            --bg-color: #f5f7fa;
            --chat-bg: #fff;
            --user-bubble: #e9effd;
            --bot-bubble: #f0f4f9;
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: var(--bg-color);
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        .app-container {
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            padding-top: 20px;
        }

        h1 {
            font-size: 2.2rem;
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1rem;
            color: #666;
            max-width: 600px;
            margin: 0 auto;
        }

        .chat-container {
            background: var(--chat-bg);
            border-radius: 12px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            height: 550px;
            flex-grow: 1;
        }

        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            position: relative;
            word-wrap: break-word;
        }

        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border-bottom-right-radius: 4px;
        }

        .bot-message {
            align-self: flex-start;
            background: var(--bot-bubble);
            color: var(--text-color);
            border-bottom-left-radius: 4px;
        }

        .chat-input-container {
            display: flex;
            padding: 15px;
            border-top: 1px solid rgba(0, 0, 0, 0.05);
        }

        #user-input {
            flex-grow: 1;
            border: none;
            background: var(--bg-color);
            padding: 12px 16px;
            border-radius: 24px;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s;
        }

        #user-input:focus {
            box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2);
        }

        #send-button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-left: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        #send-button:hover {
            transform: scale(1.05);
        }

        #send-button:active {
            transform: scale(0.95);
        }

        .loader {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        .typing-indicator {
            display: none;
            align-self: flex-start;
            background: var(--bot-bubble);
            padding: 10px 20px;
            border-radius: 20px;
            border-bottom-left-radius: 4px;
            margin-top: 10px;
        }

        .typing-indicator span {
            height: 8px;
            width: 8px;
            margin: 0 1px;
            background-color: #9E9EA1;
            display: inline-block;
            border-radius: 50%;
            opacity: 0.4;
            animation: typing 1s infinite;
        }

        .typing-indicator span:nth-child(1) {
            animation-delay: 0s;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0% {
                transform: translateY(0px);
                opacity: 0.4;
            }
            50% {
                transform: translateY(-5px);
                opacity: 0.8;
            }
            100% {
                transform: translateY(0px);
                opacity: 0.4;
            }
        }

        .model-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .control-group label {
            margin-bottom: 5px;
            font-size: 0.9rem;
            color: #666;
        }

        .control-group input {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            text-align: center;
            width: 70px;
        }

        footer {
            text-align: center;
            padding: 20px;
            margin-top: 20px;
            font-size: 0.8rem;
            color: #666;
        }

        .welcome-message {
            text-align: center;
            padding: 30px;
            color: #666;
        }

        .welcome-message p {
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .app-container {
                padding: 15px;
            }
            
            h1 {
                font-size: 1.8rem;
            }
            
            .chat-container {
                height: 450px;
            }
            
            .message {
                max-width: 90%;
            }
            
            .model-controls {
                flex-direction: column;
                gap: 10px;
            }
            
            .control-group {
                flex-direction: row;
                justify-content: space-between;
                width: 100%;
                max-width: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header>
            <h1>Hierarchical Transformer Chat</h1>
            <p class="subtitle">Interact with a block-wise hierarchical transformer model trained on conversation data</p>
        </header>

        <div class="chat-container">
            <div class="chat-history" id="chat-history">
                <div class="welcome-message">
                    <p>Welcome to the Hierarchical Transformer Chat!</p>
                    <p>This chat interface uses a custom transformer model with a hierarchical architecture that processes text in blocks for better efficiency.</p>
                    <p>Start by typing a message below to interact with the model.</p>
                </div>
            </div>
            <div class="typing-indicator" id="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="chat-input-container">
                <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
                <button id="send-button">
                    <span id="send-icon">➤</span>
                    <div class="loader" id="loader"></div>
                </button>
            </div>
        </div>

        <div class="model-controls">
            <div class="control-group">
                <label for="max-length">Max Length</label>
                <input type="number" id="max-length" value="20" min="1" max="100">
            </div>
            <div class="control-group">
                <label for="top-k">Top K</label>
                <input type="number" id="top-k" value="5" min="1" max="50">
            </div>
            <div class="control-group">
                <label for="temperature">Temperature</label>
                <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
            </div>
        </div>
    </div>

    <footer>
        Built with a Hierarchical Transformer Model | <span id="device-info">Loading...</span>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatHistory = document.getElementById('chat-history');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const sendIcon = document.getElementById('send-icon');
            const loader = document.getElementById('loader');
            const typingIndicator = document.getElementById('typing-indicator');
            const maxLengthInput = document.getElementById('max-length');
            const topKInput = document.getElementById('top-k');
            const temperatureInput = document.getElementById('temperature');
            const deviceInfo = document.getElementById('device-info');

            // Clear the welcome message on first interaction
            let welcomeCleared = false;

            // Get device info
            fetch('/device-info')
                .then(response => response.json())
                .then(data => {
                    deviceInfo.textContent = `Running on: ${data.device}`;
                })
                .catch(error => {
                    console.error('Error fetching device info:', error);
                    deviceInfo.textContent = 'Device info unavailable';
                });

            // Function to add a message to the chat history
            function addMessage(text, isUser) {
                // Clear welcome message if it's the first interaction
                if (!welcomeCleared) {
                    chatHistory.innerHTML = '';
                    welcomeCleared = true;
                }

                const messageElement = document.createElement('div');
                messageElement.classList.add('message');
                messageElement.classList.add(isUser ? 'user-message' : 'bot-message');
                messageElement.textContent = text;
                chatHistory.appendChild(messageElement);
                
                // Scroll to the bottom
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }

            // Function to show typing indicator
            function showTypingIndicator() {
                typingIndicator.style.display = 'block';
                chatHistory.scrollTop = chatHistory.scrollHeight + 50;
            }

            // Function to hide typing indicator
            function hideTypingIndicator() {
                typingIndicator.style.display = 'none';
            }

            // Function to handle sending a message
            function sendMessage() {
                const message = userInput.value.trim();
                if (message === '') return;

                // Disable input while processing
                userInput.disabled = true;
                sendIcon.style.display = 'none';
                loader.style.display = 'block';

                // Add user message to chat
                addMessage(message, true);
                
                // Show typing indicator
                showTypingIndicator();

                // Get model parameters
                const maxLength = parseInt(maxLengthInput.value) || 20;
                const topK = parseInt(topKInput.value) || 5;
                const temperature = parseFloat(temperatureInput.value) || 0.7;

                // Send request to the server
                fetch('/get-response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        max_length: maxLength,
                        top_k: topK,
                        temperature: temperature
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide typing indicator
                    hideTypingIndicator();
                    
                    // Add bot response to chat
                    addMessage(data.response, false);
                    
                    // Reset UI
                    userInput.value = '';
                    userInput.disabled = false;
                    userInput.focus();
                    sendIcon.style.display = 'block';
                    loader.style.display = 'none';
                })
                .catch(error => {
                    console.error('Error:', error);
                    hideTypingIndicator();
                    addMessage("Sorry, an error occurred. Please try again.", false);
                    userInput.disabled = false;
                    sendIcon.style.display = 'block';
                    loader.style.display = 'none';
                });
            }

            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });

            // Focus input on load
            userInput.focus();
        });
    </script>
</body>
</html>
'''

# Route for the main page
@app.route('/')
def index():
    return render_template_string(TEMPLATE)

# Route to get device info
@app.route('/device-info', methods=['GET'])
def device_info():
    device = "CPU"
    if torch.cuda.is_available():
        device = f"GPU ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "Apple Silicon (MPS)"
    return jsonify({"device": device})

# Route to get model response
@app.route('/get-response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        max_length = data.get('max_length', 20)
        top_k = data.get('top_k', 5)
        temperature = data.get('temperature', 0.7)
        
        if not user_message:
            return jsonify({"response": "Please enter a message."})
        
        # Generate response from the model
        response = get_model_response(
            user_message, 
            model, 
            device,
            max_length=max_length,
            top_k=top_k,
            temperature=temperature
        )
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in get_response: {e}")
        return jsonify({"response": "An error occurred. Please try again."})

if __name__ == '__main__':
    print("Loading model...")
    model, device = load_model()
    
    # Start the Flask application
    print("\n" + "="*50)
    print("Hierarchical Transformer Chat Server is running!")
    print("Open your browser and navigate to http://localhost:9000")
    print("="*50 + "\n")
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=9000)
