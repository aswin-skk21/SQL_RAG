from flask import Flask, render_template, request, jsonify, session
from rag import initialize_rag, get_rag_response
from langchain_core.messages import HumanMessage, AIMessage
import os

<<<<<<< HEAD
# Create a new Flask application instance
app = Flask(__name__)

# Set a secret key for securely signing the session cookie
# It tries to read the SECRET_KEY from environment variables; if not found, uses a default
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Initialize RAG (Retrieval-Augmented Generation) system once at app startup
rag_chain = None
try:
    with app.app_context():
        rag_chain = initialize_rag()  # Call your custom function to set up the RAG pipeline
except Exception as e:
    # Print error if initialization fails and set rag_chain to None
    print(f"Error initializing RAG: {e}")
    rag_chain = None

# Route for the home page that serves the main UI (likely contains chat interface)
@app.route('/', methods=['GET'])
def index():
    # Render an HTML template named 'index.html'
    return render_template('index.html')

# Route to handle chat POST requests from the frontend
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # If RAG system failed to initialize, return an error response
        if rag_chain is None:
            return jsonify({'error': 'RAG system not initialized'}), 500
            
        # Get JSON data from the incoming request (expects a 'question' field)
        data = request.get_json()
        if not data or 'question' not in data:
            # If no question is provided, return a 400 Bad Request error
=======
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Use environment variable

# Initialize RAG once when app starts
rag_chain = None
try:
    with app.app_context():
        rag_chain = initialize_rag()
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Check if RAG is initialized
        if rag_chain is None:
            return jsonify({'error': 'RAG system not initialized'}), 500
            
        data = request.get_json()
        if not data or 'question' not in data:
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
            return jsonify({'error': 'Missing question'}), 400
            
        user_question = data['question']
        
<<<<<<< HEAD
        # Retrieve the chat history stored in the user’s session; initialize if missing
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Convert the stored chat history into LangChain message objects
=======
        # Get chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Convert session history to LangChain messages
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
        chat_history = []
        for msg in session['chat_history']:
            if msg['role'] == 'human':
                chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'ai':
                chat_history.append(AIMessage(content=msg['content']))
        
<<<<<<< HEAD
        # Call your RAG response function with the question, chat history, and RAG pipeline
        answer, updated_history = get_rag_response(user_question, chat_history, rag_chain)
        
        # Convert the updated chat history back into a format storable in the session
=======
        # Call RAG function with correct parameters
        answer, updated_history = get_rag_response(user_question, chat_history, rag_chain)
        
        # Convert back to session format and update
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
        session['chat_history'] = []
        for msg in updated_history:
            if isinstance(msg, HumanMessage):
                session['chat_history'].append({'role': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                session['chat_history'].append({'role': 'ai', 'content': msg.content})
        
<<<<<<< HEAD
        # Return the generated answer as JSON to the client
        return jsonify({'answer': answer})
        
    except Exception as e:
        # If any exception occurs, return an error message and 500 status
        return jsonify({'error': str(e)}), 500

# Run the Flask development server on port 5001 with debug mode enabled
=======
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
if __name__ == '__main__':
    app.run(debug=True, port=5001)