from flask import Flask, render_template, request, jsonify 
from rag import initialize_rag, get_rag_response

app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/chat', methods=['POST'])

def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)