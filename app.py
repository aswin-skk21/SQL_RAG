from flask import Flask, render_template, request, jsonify 
import rag.py as rag

app = Flask(__name__)

@app.route('/')
@app.route('/chat')

def index():
    return "hello world"

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/get', methods=['GET', 'POST'])

def get_rag_response():
    