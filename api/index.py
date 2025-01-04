from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import nltk
import pdfplumber
import random
import firebase_admin
from firebase_admin import credentials, auth
import PyPDF2
from io import BytesIO

app = Flask(__name__, template_folder='../templates')
CORS(app)

# تهيئة Firebase
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")

# تحميل موارد NLTK
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
except Exception as e:
    print(f"NLTK download error: {str(e)}")

def generate_question_from_sentence(sentence, question_type='essay'):
    """Generate a question from a given sentence."""
    try:
        if question_type == 'mcq':
            # توليد سؤال اختيار من متعدد
            question = f"Which of the following best describes: {sentence}?"
            # توليد خيارات عشوائية بسيطة
            correct_answer = sentence
            wrong_answers = [
                f"Not: {sentence}",
                f"Maybe: {sentence}",
                f"Opposite of: {sentence}"
            ]
            choices = [correct_answer] + wrong_answers
            random.shuffle(choices)
            correct_index = choices.index(correct_answer)
            
            return {
                'question': question,
                'choices': choices,
                'correct_answer': correct_index
            }
        else:
            # توليد سؤال مقالي
            return {
                'question': f"Explain the following: {sentence}",
                'answer': sentence
            }
    except Exception as e:
        print(f"Error generating question: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # استخراج النص من الملف
        pdf_text = extract_text_from_pdf(file)
        if not pdf_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400

        # تقسيم النص إلى جمل
        sentences = nltk.sent_tokenize(pdf_text)
        
        # الحصول على عدد الأسئلة المطلوب وتحديد النوع
        num_questions = int(request.form.get('num_questions', 5))
        question_type = request.form.get('question_type', 'essay')
        
        # اختيار جمل عشوائية وتوليد أسئلة منها
        selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
        questions = []
        
        for sentence in selected_sentences:
            question = generate_question_from_sentence(sentence, question_type)
            if question:
                questions.append(question)

        return jsonify({'questions': questions}), 200

    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
