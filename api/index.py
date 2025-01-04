from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
import os

app = Flask(__name__, 
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)