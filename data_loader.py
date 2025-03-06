"""
Data Loader for Hierarchical Transformer Model
Handles loading, processing, and preparing conversation data from various sources.
"""

import os
import json
import random
import time
import requests
import re
import csv
from typing import List, Dict, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataLoader")

# Base protocol for data sources
class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    @abstractmethod
    def get_data(self) -> List[List[str]]:
        """Retrieve conversation data as a list of conversation turns"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data source"""
        pass
    
    def info(self) -> Dict:
        """Return information about this data source"""
        return {
            "name": self.name(),
            "type": self.__class__.__name__
        }

# Local file data sources
class JSONDataSource(DataSource):
    """Loads conversation data from a JSON file"""
    
    def __init__(self, file_path: str, conversation_key: str = None):
        """
        Initialize with file path and optional key to access conversations in the JSON
        
        Args:
            file_path: Path to the JSON file
            conversation_key: Optional key to access the conversation list in the JSON
        """
        self.file_path = file_path
        self.conversation_key = conversation_key
        
    def name(self) -> str:
        return f"JSON({os.path.basename(self.file_path)})"
    
    def get_data(self) -> List[List[str]]:
        """Load and parse conversations from JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.conversation_key:
                if self.conversation_key not in data:
                    logger.warning(f"Key '{self.conversation_key}' not found in JSON")
                    return []
                conversations = data[self.conversation_key]
            else:
                conversations = data
                
            return self._process_conversations(conversations)
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            return []
    
    def _process_conversations(self, conversations) -> List[List[str]]:
        """Process the loaded JSON data into the expected format"""
        results = []
        
        # Handle different JSON formats
        if isinstance(conversations, list):
            for conv in conversations:
                if isinstance(conv, list):
                    # Already in the expected format
                    if all(isinstance(turn, str) for turn in conv):
                        results.append(conv)
                elif isinstance(conv, dict):
                    # Extract conversation turns from dict
                    if 'turns' in conv and isinstance(conv['turns'], list):
                        results.append(conv['turns'])
                    elif 'messages' in conv and isinstance(conv['messages'], list):
                        # Extract just the content from messages
                        turns = []
                        for msg in conv['messages']:
                            if isinstance(msg, dict) and 'content' in msg:
                                turns.append(msg['content'])
                            elif isinstance(msg, str):
                                turns.append(msg)
                        if turns:
                            results.append(turns)
        
        return results

class CSVDataSource(DataSource):
    """Loads conversation data from a CSV file"""
    
    def __init__(self, file_path: str, 
                 question_col: str = "question", 
                 answer_col: str = "answer",
                 delimiter: str = ','):
        """
        Initialize with file path and column names
        
        Args:
            file_path: Path to the CSV file
            question_col: Column name for questions/prompts
            answer_col: Column name for answers/responses
            delimiter: CSV delimiter character
        """
        self.file_path = file_path
        self.question_col = question_col
        self.answer_col = answer_col
        self.delimiter = delimiter
        
    def name(self) -> str:
        return f"CSV({os.path.basename(self.file_path)})"
    
    def get_data(self) -> List[List[str]]:
        """Load and parse conversations from CSV file"""
        try:
            conversations = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                
                for row in reader:
                    if self.question_col in row and self.answer_col in row:
                        question = row[self.question_col].strip()
                        answer = row[self.answer_col].strip()
                        if question and answer:  # Skip empty entries
                            conversations.append([question, answer])
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return []

class TXTDataSource(DataSource):
    """Loads conversation data from a text file with a specific format"""
    
    def __init__(self, file_path: str, separator: str = "==="):
        """
        Initialize with file path and separator
        
        Args:
            file_path: Path to the text file
            separator: String that separates different conversations
        """
        self.file_path = file_path
        self.separator = separator
        
    def name(self) -> str:
        return f"TXT({os.path.basename(self.file_path)})"
    
    def get_data(self) -> List[List[str]]:
        """Load and parse conversations from text file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into conversations
            raw_conversations = content.split(self.separator)
            
            conversations = []
            for conv in raw_conversations:
                # Clean and split into turns
                turns = [t.strip() for t in conv.strip().split('\n') if t.strip()]
                if turns:
                    conversations.append(turns)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            return []

# Remote data sources
class WebAPIDataSource(DataSource):
    """Loads conversation data from a web API"""
    
    def __init__(self, 
                 url: str, 
                 headers: Dict = None, 
                 params: Dict = None,
                 extract_path: str = None,
                 data_processor: Callable = None):
        """
        Initialize with API details
        
        Args:
            url: API endpoint URL
            headers: HTTP headers for the request
            params: Query parameters for the request
            extract_path: JSON path to extract conversations (dot notation)
            data_processor: Optional function to process the raw API response
        """
        self.url = url
        self.headers = headers or {}
        self.params = params or {}
        self.extract_path = extract_path
        self.data_processor = data_processor
        
    def name(self) -> str:
        return f"WebAPI({self.url.split('://')[1].split('/')[0]})"
    
    def get_data(self) -> List[List[str]]:
        """Fetch and parse conversations from web API"""
        try:
            response = requests.get(
                self.url, 
                headers=self.headers,
                params=self.params,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}")
                return []
            
            data = response.json()
            
            # Extract data if path is specified
            if self.extract_path:
                for key in self.extract_path.split('.'):
                    if key in data:
                        data = data[key]
                    else:
                        logger.warning(f"Key '{key}' not found in API response")
                        return []
            
            # Apply custom processor if provided
            if self.data_processor:
                return self.data_processor(data)
            
            # Otherwise, try to interpret as a list of conversations
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list) and all(isinstance(turn, str) for turn in item):
                        result.append(item)
                    elif isinstance(item, dict) and 'conversation' in item:
                        conv = item['conversation']
                        if isinstance(conv, list):
                            result.append(conv)
                return result
            
            logger.warning("Couldn't interpret API response as conversation data")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return []

# Synthetic data generation
class SyntheticDataSource(DataSource):
    """Generates synthetic conversation data based on templates"""
    
    def __init__(self, templates: List[Tuple[str, str]], count: int = 100):
        """
        Initialize with templates and count
        
        Args:
            templates: List of (prompt_template, response_template) pairs
            count: Number of synthetic conversations to generate
        """
        self.templates = templates
        self.count = count
        
        # Common placeholders for templates
        self.placeholders = {
            "{name}": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah"],
            "{color}": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black"],
            "{food}": ["pizza", "pasta", "burger", "salad", "sushi", "tacos", "sandwich", "cake"],
            "{animal}": ["dog", "cat", "elephant", "lion", "tiger", "giraffe", "monkey", "bear"],
            "{number}": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "{day}": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "{city}": ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Rome", "Cairo"],
            "{country}": ["USA", "UK", "Japan", "France", "Germany", "Australia", "Italy", "Egypt"],
        }
    
    def name(self) -> str:
        return f"Synthetic({self.count})"
    
    def get_data(self) -> List[List[str]]:
        """Generate synthetic conversation data"""
        conversations = []
        
        for _ in range(self.count):
            template_idx = random.randint(0, len(self.templates) - 1)
            prompt_template, response_template = self.templates[template_idx]
            
            # Fill in placeholders
            prompt = self._fill_template(prompt_template)
            response = self._fill_template(response_template)
            
            conversations.append([prompt, response])
        
        return conversations
    
    def _fill_template(self, template: str) -> str:
        """Replace placeholders in template with random values"""
        result = template
        
        # Find all placeholders in the template
        placeholders = re.findall(r"(\{[a-z_]+\})", template)
        
        # Replace each placeholder
        for placeholder in placeholders:
            if placeholder in self.placeholders:
                replacement = random.choice(self.placeholders[placeholder])
                result = result.replace(placeholder, replacement)
        
        return result

# Data processor for cleaning and combining data
class DataProcessor:
    """Processes, cleans, and transforms conversation data"""
    
    def __init__(self):
        self.sources = []
        self.stats = {"total_conversations": 0, "total_turns": 0, "source_stats": {}}
    
    def add_source(self, source: DataSource) -> None:
        """Add a data source"""
        self.sources.append(source)
        logger.info(f"Added data source: {source.name()}")
    
    def process_all_sources(self) -> List[List[str]]:
        """Process all data sources and return combined conversations"""
        all_conversations = []
        self.stats = {"total_conversations": 0, "total_turns": 0, "source_stats": {}}
        
        for source in self.sources:
            start_time = time.time()
            logger.info(f"Processing source: {source.name()}")
            
            conversations = source.get_data()
            cleaned_conversations = self._clean_conversations(conversations)
            
            # Update stats
            source_name = source.name()
            self.stats["source_stats"][source_name] = {
                "raw_count": len(conversations),
                "cleaned_count": len(cleaned_conversations),
                "time_taken": round(time.time() - start_time, 2)
            }
            
            all_conversations.extend(cleaned_conversations)
            logger.info(f"Added {len(cleaned_conversations)} conversations from {source.name()}")
        
        # Overall stats
        self.stats["total_conversations"] = len(all_conversations)
        self.stats["total_turns"] = sum(len(conv) for conv in all_conversations)
        
        return all_conversations
    
    def _clean_conversations(self, conversations: List[List[str]]) -> List[List[str]]:
        """Clean and filter conversations"""
        cleaned = []
        
        for conv in conversations:
            # Skip empty conversations
            if not conv:
                continue
                
            # Clean each turn
            cleaned_turns = []
            for turn in conv:
                # Basic cleaning
                clean_turn = self._clean_text(turn)
                if clean_turn:
                    cleaned_turns.append(clean_turn)
            
            # Only add if we have at least 2 turns
            if len(cleaned_turns) >= 2:
                cleaned.append(cleaned_turns)
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text turns"""
        if not text:
            return ""
            
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'[^\w\s.,?!\'"-]', '', text)  # Remove special characters except basics
        
        # Limit length
        if len(text) > 512:
            text = text[:512]
            
        return text
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats
    
    def save_processed_data(self, file_path: str) -> None:
        """Save processed conversations to a file"""
        all_conversations = self.process_all_sources()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_conversations, f, indent=2)
            logger.info(f"Saved {len(all_conversations)} conversations to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")

# Helper functions for common templates
def get_conversation_templates() -> List[Tuple[str, str]]:
    """Return common conversation templates for synthetic data"""
    return [
        # Greetings
        ("hello", "hi there, how are you?"),
        ("hi", "hello, how can I help you today?"),
        ("good morning", "good morning! How are you today?"),
        ("how are you", "I'm doing well, thank you for asking! How about you?"),
        
        # Questions
        ("what is your name", "my name is Assistant, nice to meet you!"),
        ("how old are you", "I exist as a language model, so I don't have an age in the traditional sense."),
        ("what can you do", "I can chat with you, answer questions, help with tasks, and more!"),
        ("tell me about yourself", "I'm an AI assistant designed to be helpful, harmless, and honest."),
        
        # Template-based
        ("what is your favorite {color}", "I like all colors, but {color} is particularly nice!"),
        ("do you like {food}", "Yes, {food} is delicious! Do you enjoy it too?"),
        ("have you been to {city}", "I haven't been to {city}, but I've heard it's a wonderful place!"),
        ("tell me about {animal}s", "{animal}s are fascinating creatures with unique characteristics."),
        
        # Instructions
        ("how do I make {food}", "To make {food}, you'll need ingredients and follow specific steps..."),
        ("what's the weather in {city}", "I don't have real-time weather data for {city}, but you can check online."),
        ("how many {animal}s are there in the world", "There are many {animal}s in the world, though exact numbers vary."),
        
        # Opinions/Preferences
        ("what do you think about {topic}", "That's an interesting question about {topic}. There are multiple perspectives..."),
        ("is {food} healthy", "{food} can be part of a balanced diet, depending on how it's prepared."),
        ("which is better, {color} or {color}", "Both colors have their unique appeal, it's really a matter of personal preference!"),
    ]

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Add synthetic data source
    templates = get_conversation_templates()
    synthetic = SyntheticDataSource(templates, count=1000)
    processor.add_source(synthetic)
    
    # Process all data
    conversations = processor.process_all_sources()
    
    # Print stats
    print(json.dumps(processor.get_stats(), indent=2))
    
    # Save to file
    processor.save_processed_data("enhanced_conversations.json")
