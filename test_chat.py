"""
Test script for the hierarchical transformer chat application.
This script sends a test message to the chat server and prints the response.
"""

import requests
import json
import sys
import time
import argparse

def test_chat_server(host="localhost", port=9000, message="Hello, how are you?"):
    """Test the chat server by sending a message and displaying the response."""
    url = f"http://{host}:{port}/get-response"
    
    # Get device info first
    try:
        device_response = requests.get(f"http://{host}:{port}/device-info")
        if device_response.status_code == 200:
            device_info = device_response.json()
            print(f"Server is running on: {device_info.get('device', 'Unknown device')}")
        else:
            print(f"Failed to get device info. Status code: {device_response.status_code}")
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        print(f"Make sure the server is running at http://{host}:{port}")
        return
    
    # Prepare the request payload
    payload = {
        "message": message,
        "max_length": 20,
        "top_k": 5,
        "temperature": 0.7
    }
    
    # Send the request and measure response time
    print(f"\nSending test message: '{message}'")
    print("Waiting for response...")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            response_data = response.json()
            model_response = response_data.get("response", "No response")
            
            print("\n" + "="*50)
            print(f"Response received in {end_time - start_time:.2f} seconds:")
            print(f"Model: {model_response}")
            print("="*50)
            
            # Provide some analysis of the response
            if model_response == "Sorry, I had trouble processing that. Can you try again?":
                print("\nThe model encountered an error. Check server logs for details.")
            elif not model_response or model_response.strip() == "":
                print("\nThe model returned an empty response.")
            else:
                word_count = len(model_response.split())
                print(f"\nResponse contains {word_count} words.")
                print("Server appears to be functioning correctly!")
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)
    except requests.RequestException as e:
        print(f"Error communicating with server: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the hierarchical transformer chat server")
    parser.add_argument("--host", default="localhost", help="Host where the chat server is running")
    parser.add_argument("--port", type=int, default=9000, help="Port where the chat server is running")
    parser.add_argument("--message", default="Hello, how are you?", help="Test message to send to the chat server")
    
    args = parser.parse_args()
    test_chat_server(args.host, args.port, args.message)
