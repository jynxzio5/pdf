from flask import Flask, render_template, request, jsonify
import pdfplumber
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import uuid
import json
from flask_cors import CORS

try:
    # Initialize Firebase with explicit path
    service_account_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serviceAccountKey.json')
    
    # Load and validate the service account file
    with open(service_account_path, 'r') as f:
        service_account_info = json.load(f)
    
    cred = credentials.Certificate(service_account_info)
    
    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    raise e

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # إضافة دعم CORS للسماح بطلبات المصادقة
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def generate_question_from_sentence(sentence, question_type='essay'):
    """Generate a question from a given sentence."""
    # Remove common Arabic question words if they exist at the start
    question_words = ['ما', 'ماذا', 'كيف', 'لماذا', 'متى', 'أين', 'من', 'هل']
    for word in question_words:
        if sentence.startswith(word):
            sentence = sentence[len(word):].strip()
    
    if question_type == 'multiple_choice':
        # For multiple choice, we'll create variations of the sentence
        words = word_tokenize(sentence)
        important_words = [w for w in words if len(w) > 3 and w not in stopwords.words('arabic')]
        
        if not important_words:
            return {
                'question': f'اختر الإجابة الصحيحة: {sentence}',
                'options': [
                    sentence,
                    f'ليس {sentence}',
                    f'نعم، {sentence}',
                    f'لا، {sentence}'
                ],
                'correct_answer': sentence
            }
        
        # Create wrong options by replacing important words
        options = [sentence]
        for i in range(3):
            if important_words:
                word_to_replace = random.choice(important_words)
                wrong_option = sentence.replace(word_to_replace, f'[كلمة بديلة {i+1}]')
                options.append(wrong_option)
        
        return {
            'question': f'اختر الإجابة الصحيحة: {sentence}',
            'options': options,
            'correct_answer': sentence
        }
    else:
        # For essay questions, we'll create reflective questions
        question_starters = [
            'ما رأيك في',
            'اشرح بالتفصيل',
            'حلل العبارة التالية',
            'قارن بين',
            'ما هي أهمية'
        ]
        starter = random.choice(question_starters)
        return {
            'question': f'{starter}: {sentence}',
            'answer': 'اكتب إجابتك هنا...'
        }

def extract_and_generate_questions(text, question_type='essay', num_questions=5):
    """Extract sentences and generate questions."""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Filter out very short sentences and those that don't end with proper punctuation
    valid_sentences = [s for s in sentences if len(s) > 20 and any(s.endswith(p) for p in ['.', '؟', '!'])]
    
    # Select random sentences for questions
    selected_sentences = random.sample(valid_sentences, min(num_questions, len(valid_sentences)))
    
    # Generate questions
    questions = []
    for sentence in selected_sentences:
        question = generate_question_from_sentence(sentence, question_type)
        questions.append(question)
    
    return questions

def save_to_firebase(questions, original_filename):
    try:
        # Create a new document with a unique ID
        doc_id = str(uuid.uuid4())
        doc_ref = db.collection('questions').document(doc_id)
        
        # Convert datetime to string to make it JSON serializable
        current_time = datetime.datetime.now().isoformat()
        
        # Prepare the document data
        doc_data = {
            'id': doc_id,
            'questions': questions,
            'original_filename': original_filename,
            'created_at': current_time,
            'updated_at': current_time
        }
        
        # Save to Firestore
        doc_ref.set(doc_data)
        return doc_id
    except Exception as e:
        print(f"Error saving to Firebase: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth/callback')
def auth_callback():
    return jsonify({'status': 'success'})

@app.route('/questions/history', methods=['GET'])
def get_questions_history():
    try:
        # Get all documents from the questions collection
        docs = db.collection('questions').stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                'id': data['id'],
                'original_filename': data['original_filename'],
                'created_at': data['created_at']
            })
        
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error getting history: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء تحميل السجل'}), 500

@app.route('/questions/<document_id>', methods=['GET'])
def get_questions(document_id):
    try:
        doc_ref = db.collection('questions').document(document_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({'error': 'لم يتم العثور على الأسئلة'}), 404
    except Exception as e:
        print(f"Error getting questions: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء استرجاع البيانات'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم تحديد ملف'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the file
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + ' '
            
            questions = extract_and_generate_questions(text)
            
            # Save to Firebase
            doc_id = save_to_firebase(questions, file.filename)
            
            # Clean up
            os.remove(file_path)
            
            if doc_id:
                return jsonify({
                    'result': questions,
                    'document_id': doc_id
                })
            else:
                return jsonify({'error': 'حدث خطأ أثناء حفظ البيانات'}), 500
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': 'حدث خطأ أثناء معالجة الملف'}), 500
    else:
        return jsonify({'error': 'يجب رفع ملف PDF فقط'}), 400

if __name__ == '__main__':
    app.run(debug=True)
