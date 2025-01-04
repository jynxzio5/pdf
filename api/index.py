from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import nltk
import pdfplumber
import random
import PyPDF2
from io import BytesIO

# تعيين مسار NLTK
nltk.data.path.append("/tmp/nltk_data")

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# تحميل موارد NLTK فقط إذا لم تكن موجودة
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir="/tmp/nltk_data")

# تحميل البيانات عند بدء التطبيق
download_nltk_data()

def generate_question_from_sentence(sentence, question_type='essay'):
    """توليد سؤال من جملة معينة."""
    try:
        sentence = sentence.strip()
        if len(sentence) < 10:  # تجاهل الجمل القصيرة جداً
            return None

        if question_type == 'mcq':
            # توليد سؤال اختيار من متعدد
            question = f"اختر الإجابة الصحيحة: {sentence}?"
            correct_answer = sentence
            
            # توليد إجابات خاطئة
            words = sentence.split()
            if len(words) > 3:
                wrong_answers = [
                    ' '.join(words[:-1] + [random.choice(words)]),
                    ' '.join([random.choice(words)] + words[1:]),
                    ' '.join(reversed(words))
                ]
            else:
                wrong_answers = [
                    "ليست الإجابة الصحيحة",
                    "إجابة غير صحيحة",
                    "خيار خاطئ"
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
                'question': f"اشرح التالي: {sentence}",
                'answer': sentence
            }
    except Exception as e:
        print(f"خطأ في توليد السؤال: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """استخراج النص من ملف PDF."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text
    except Exception as e:
        print(f"خطأ في استخراج النص: {str(e)}")
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'يُسمح فقط بملفات PDF'}), 400

        # التحقق من حجم الملف (16MB كحد أقصى)
        file_content = file.read()
        if len(file_content) > 16 * 1024 * 1024:  # 16MB in bytes
            return jsonify({'error': 'حجم الملف يتجاوز الحد المسموح به (16MB)'}), 400

        # إعادة تعيين مؤشر الملف
        file = BytesIO(file_content)

        # التحقق من صحة ملف PDF
        try:
            PyPDF2.PdfReader(file)
            file.seek(0)
        except Exception as e:
            return jsonify({'error': 'الملف المرفوع ليس ملف PDF صالح'}), 400

        # استخراج النص
        text = extract_text_from_pdf(file)
        if not text.strip():
            return jsonify({'error': 'لم يتم العثور على نص قابل للاستخراج في الملف'}), 400

        # تقسيم النص إلى جمل
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return jsonify({'error': 'لم يتم العثور على جمل كافية في النص'}), 400

        # تنظيف وفلترة الجمل
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return jsonify({'error': 'لم يتم العثور على جمل مناسبة لتوليد الأسئلة'}), 400

        # الحصول على المعلمات
        try:
            num_questions = min(max(1, int(request.form.get('num_questions', 5))), 10)
        except ValueError:
            num_questions = 5

        question_type = request.form.get('question_type', 'essay')
        if question_type not in ['essay', 'mcq']:
            question_type = 'essay'

        # توليد الأسئلة
        selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
        questions = []
        
        for sentence in selected_sentences:
            question = generate_question_from_sentence(sentence, question_type)
            if question:
                questions.append(question)

        if not questions:
            return jsonify({'error': 'لم نتمكن من توليد أسئلة من النص المقدم'}), 400

        return jsonify({'questions': questions}), 200

    except Exception as e:
        print(f"خطأ في generate_questions: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء معالجة الملف'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
