# SQL RAG Assistant

A Flask-based RAG (Retrieval-Augmented Generation) application that provides SQL assistance using Google's Gemini model.

## Features

- 🤖 SQL query generation from natural language
- 💬 Interactive chat interface
- 🔍 RAG-powered responses using your database schema
- 📱 Responsive design
- 🚀 Deployed on Vercel

## Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SQL_RAG
   ```

2. **Set up virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Visit the application**
   Open http://localhost:5001 in your browser

## Deployment on Vercel

### Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **Google API Key**: Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Deployment Steps

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy the application**
   ```bash
   vercel
   ```

3. **Set environment variables in Vercel**
   - Go to your Vercel dashboard
   - Select your project
   - Go to Settings → Environment Variables
   - Add:
     - `GOOGLE_API_KEY`: Your Google API key
     - `SECRET_KEY`: A random secret key for sessions

4. **Redeploy with environment variables**
   ```bash
   vercel --prod
   ```

### Important Notes for Vercel Deployment

- **Vector Store**: The vector store will be recreated on each deployment
- **Cold Starts**: First request may take longer due to embedding creation
- **Memory Limits**: Vercel has memory limits, so large datasets may need optimization
- **Session Storage**: Sessions are ephemeral on Vercel (serverless)

## Project Structure

```
SQL_RAG/
├── app.py                 # Main Flask application
├── rag.py                 # RAG system implementation
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel configuration
├── runtime.txt           # Python runtime version
├── .env                  # Environment variables (local)
├── .gitignore           # Git ignore rules
├── templates/
│   └── index.html       # Main HTML template
└── static/
    ├── css/
    │   └── style.css    # Styles
    └── js/
        └── chat.js      # Frontend JavaScript
```

## Environment Variables

- `GOOGLE_API_KEY`: Your Google AI API key
- `SECRET_KEY`: Secret key for Flask sessions

## Troubleshooting

### Common Issues

1. **Port 5000 in use**: Change port in `app.py` or disable AirPlay Receiver
2. **ChromaDB errors**: Delete `chroma_langchain_db/` folder and restart
3. **API key errors**: Ensure `GOOGLE_API_KEY` is set correctly
4. **Memory issues**: Consider using smaller datasets for Vercel deployment

### Vercel-Specific Issues

1. **Build failures**: Check `requirements.txt` for compatibility
2. **Cold start delays**: First request will be slower due to embedding creation
3. **Session issues**: Sessions are ephemeral on serverless platforms

## License

MIT License
