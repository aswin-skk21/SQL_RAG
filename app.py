from flask import Flask, render_template, request, jsonify
from rag import initialize_rag, get_rag_response

app = Flask(__name__)
rag_chain = initialize_rag()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({'error': 'Missing user_input'}), 400
            
        user_input = data['user_input']
        
        # Pass the rag_chain to your function if needed
        response = get_rag_response(rag_chain, user_input)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)