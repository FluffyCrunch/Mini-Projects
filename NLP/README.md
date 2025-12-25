# Real-Time Search Engine using NLP

A Natural Language Processing project that implements a real-time search engine powered by Google Search API and Groq's LLM (Llama3-70b). The system provides intelligent, context-aware responses by combining web search results with advanced language model capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Keys](#api-keys)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project creates an intelligent search engine that:
- Performs real-time Google searches based on user queries
- Uses Groq's Llama3-70b model to generate contextual responses
- Maintains conversation history in JSON format
- Provides real-time date and time information
- Delivers professional, grammatically correct answers

## ✨ Features

- **Real-Time Web Search**: Integrates Google Search API for up-to-date information
- **AI-Powered Responses**: Uses Groq's Llama3-70b model for intelligent responses
- **Conversation Memory**: Maintains chat history in JSON format
- **Context Awareness**: Combines search results with real-time information
- **Streaming Responses**: Provides real-time streaming output
- **Professional Formatting**: Ensures proper grammar and punctuation
- **Interactive CLI**: Simple command-line interface for queries

## 🏗️ Architecture

The system follows this workflow:

1. **User Query Input**: Receives user query via command line
2. **Google Search**: Performs web search and retrieves top 5 results
3. **Context Building**: Combines search results with:
   - System instructions
   - Real-time date/time information
   - Conversation history
4. **LLM Processing**: Sends context to Groq API (Llama3-70b)
5. **Response Generation**: Streams and formats the response
6. **History Update**: Saves conversation to JSON file

## 📦 Requirements

- Python 3.7+
- Google Search API (via `googlesearch` library)
- Groq API key
- Environment variables for configuration

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/FluffyCrunch/Mini-Projects.git
cd Mini-Projects/NLP
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install googlesearch-python groq python-dotenv
```

## ⚙️ Configuration

### Step 1: Create `.env` file

Create a `.env` file in the `NLP` folder with the following variables:

```env
Username=YourName
Assistantname=YourAssistantName
GroqAPIKey=your_groq_api_key_here
```

### Step 2: Get API Keys

1. **Groq API Key**:
   - Visit https://console.groq.com/
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new API key
   - Copy and paste it in your `.env` file

2. **Google Search**:
   - The `googlesearch-python` library is used (no API key required)
   - Note: Rate limits may apply with heavy usage

## 🚀 Usage

### Basic Usage

1. Make sure your `.env` file is configured with API keys
2. Run the script:
```bash
python RealTimeSearchEngine.py
```

3. Enter your queries when prompted:
```
Enter your query: What is the weather today?
Enter your query: Tell me about machine learning
Enter your query: exit  # to quit
```

### Using as a Module

You can also import and use the functions:

```python
from RealTimeSearchEngine import RealtimeSearchEngine

# Get a response
response = RealtimeSearchEngine("What is artificial intelligence?")
print(response)
```

## 📁 Project Structure

```
NLP/
├── RealTimeSearchEngine.py    # Main implementation file
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
├── .gitignore                  # Git ignore rules
└── Data/
    └── ChatLog.json            # Conversation history
```

## 🔬 How It Works

### 1. Query Processing
- User enters a query
- System loads conversation history from JSON

### 2. Web Search
- Performs Google search with the query
- Retrieves top 5 results with titles and descriptions
- Formats results for LLM context

### 3. Context Building
- Combines:
  - System instructions (personality, formatting rules)
  - Google search results
  - Real-time date/time information
  - Previous conversation history

### 4. LLM Generation
- Sends context to Groq API (Llama3-70b model)
- Receives streaming response
- Processes and cleans the output

### 5. Response & Storage
- Displays formatted response
- Saves conversation to `Data/ChatLog.json`
- Maintains context for future queries

## 🔑 API Keys

### Groq API
- **Model Used**: `llama3-70b-8192`
- **Features**: 
  - Streaming responses
  - Temperature: 0.7 (balanced creativity)
  - Max tokens: 2048
  - Top-p: 1

### Google Search
- Uses `googlesearch-python` library
- Retrieves top 5 results per query
- No API key required (but rate limits apply)

## 📝 Notes

- **Rate Limits**: Be aware of API rate limits for both Google Search and Groq
- **Privacy**: Chat logs are stored locally in JSON format
- **Internet Required**: Requires active internet connection for searches
- **API Costs**: Groq API may have usage limits (check their pricing)

## 🔮 Future Improvements

- [ ] Add web interface (Flask/FastAPI)
- [ ] Implement rate limiting and error handling
- [ ] Add support for multiple LLM providers
- [ ] Implement conversation export/import
- [ ] Add query history search
- [ ] Support for file uploads and document analysis
- [ ] Add voice input/output capabilities
- [ ] Implement caching for frequently asked questions
- [ ] Add support for multiple languages
- [ ] Create API endpoints for integration

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Error**:
   - Check your `.env` file exists
   - Verify API key is correct
   - Ensure no extra spaces in `.env` file

3. **Google Search Rate Limit**:
   - Wait a few seconds between queries
   - Consider using official Google Search API for production

4. **File Not Found Error**:
   - Ensure `Data/ChatLog.json` directory exists
   - The script will create it automatically if missing

## 👤 Author

FluffyCrunch

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This is a mini-project for learning purposes. For production use, consider implementing proper error handling, rate limiting, and security measures.

