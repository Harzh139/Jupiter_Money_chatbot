# Jupiter Money QA Bot 🏦

A conversational AI assistant for Jupiter Money customer support, built with Streamlit, Groq API, and FAISS vector search.

## Features

- 💬 Interactive chat interface
- 🔍 Semantic search through Jupiter Money documentation
- 📊 Analytics dashboard with confidence scoring
- 🔄 LangSmith integration for tracing
- 💾 Persistent FAISS vector storage

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B)
- **Vector Store**: FAISS with Sentence Transformers
- **Web Scraping**: BeautifulSoup4
- **Monitoring**: LangSmith

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd jupiter_qa_bot
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
LANGCHAIN_API_KEY = "your_langchain_api_key_here"  # Optional
LANGCHAIN_PROJECT = "JupiterBot"  # Optional
```

### 3. Initialize Data

```bash
# Scrape Jupiter Money website
python improved_scraper.py

# Create vector embeddings
python vector_store.py

# Test the QA bot
python llm_handler.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard
4. Deploy!

### Docker Deployment

```bash
docker build -t jupiter-qa-bot .
docker run -p 8501:8501 jupiter-qa-bot
```

### Local Development

```bash
streamlit run app.py --server.port 8501
```

## Project Structure

```
jupiter_qa_bot/
├── app.py                 # Main Streamlit application
├── llm_handler.py         # Groq API integration
├── vector_store.py        # FAISS vector database
├── improved_scraper.py    # Web scraping logic
├── requirements.txt       # Python dependencies
├── data/                  # Scraped and processed data
├── faiss_db/             # FAISS index storage
└── README.md             # This file
```

## Usage

1. **Ask Questions**: Type questions about Jupiter Money in the chat
2. **View Analytics**: Check the Analytics tab for performance metrics
3. **Confidence Scores**: Each answer includes a confidence rating
4. **Source Tracking**: See which documents were used for answers

## Sample Questions

- "What is Jupiter Money?"
- "How do I open an account?"
- "What are the main features?"
- "How can I contact support?"
- "What are the fees and charges?"

## Configuration

### Model Settings
- **LLM Model**: `llama-3.3-70b-versatile`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Vector Search**: FAISS IndexFlatL2
- **Temperature**: 0.3 (focused responses)

### Performance
- **Response Time**: ~2-5 seconds
- **Embedding Dimensions**: 384
- **Search Results**: Top 5 relevant chunks
- **Confidence Threshold**: 0.3+ for relevant results

## Troubleshooting

### Common Issues

1. **"No data loaded"**
   - Run `python improved_scraper.py` first

2. **"GROQ_API_KEY not found"**
   - Add API key to `.streamlit/secrets.toml`

3. **"No FAISS index available"**
   - Run `python vector_store.py` to create embeddings

4. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for assessment purposes.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.
