from flask import Flask, render_template, request, jsonify, session
from rag import initialize_rag, get_rag_response
from langchain_core.messages import HumanMessage, AIMessage
import os

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
            return jsonify({'error': 'Missing question'}), 400
            
        user_question = data['question']
        
        # Retrieve the chat history stored in the user’s session; initialize if missing
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Convert the stored chat history into LangChain message objects
        chat_history = []
        for msg in session['chat_history']:
            if msg['role'] == 'human':
                chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'ai':
                chat_history.append(AIMessage(content=msg['content']))
        
        # Call your RAG response function with the question, chat history, and RAG pipeline
        answer, updated_history = get_rag_response(user_question, chat_history, rag_chain)
        
        # Convert the updated chat history back into a format storable in the session
        session['chat_history'] = []
        for msg in updated_history:
            if isinstance(msg, HumanMessage):
                session['chat_history'].append({'role': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                session['chat_history'].append({'role': 'ai', 'content': msg.content})
        
        # Return the generated answer as JSON to the client
        return jsonify({'answer': answer})
        
    except Exception as e:
        # If any exception occurs, return an error message and 500 status
        return jsonify({'error': str(e)}), 500

# Run the Flask development server on port 5001 with debug mode enabled
if __name__ == '__main__':
    app.run(debug=True, port=5001)