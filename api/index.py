from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pdfplumber
import PyPDF2
import re
from transformers import pipeline, AutoModelForSeq2SeqGeneration, AutoTokenizer
import torch
import random

app = Flask(__name__, 
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
)
CORS(app)

# تهيئة نموذج توليد الأسئلة
model_name = "d4data/arabic-t5-base-question-generation"
tokenizer = None
model = None
question_generator = None

def load_model():
    global tokenizer, model, question_generator
    if question_generator is None:
        try:
            print("بدء تحميل النموذج...")
            # تعيين المسار للتخزين المؤقت
            cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # تحميل النموذج مع تحديد مسار التخزين المؤقت
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name, cache_dir=cache_dir)
            
            # تحديد الجهاز المناسب
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            question_generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1  # استخدام CPU دائماً لتجنب مشاكل الذاكرة
            )
            print("تم تحميل النموذج بنجاح!")
            return True
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {str(e)}")
            return False
    return True

def is_valid_pdf(file):
    """التحقق من صحة ملف PDF"""
    try:
        PyPDF2.PdfReader(file)
        return True
    except:
        return False

def extract_text_from_pdf(file):
    """استخراج النص من ملف PDF"""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def clean_text(text):
    """تنظيف النص المستخرج"""
    # إزالة الأسطر الجديدة المتعددة
    text = re.sub(r'\n+', ' ', text)
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    # تنظيف علامات الترقيم
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text.strip()

def split_text_into_chunks(text, max_length=512):
    """تقسيم النص إلى أجزاء صغيرة"""
    sentences = re.split('[.!؟\n]', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if current_length + len(sentence) + 1 <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_questions_from_text(text, num_questions=5, question_type='mcq'):
    """توليد أسئلة باستخدام نموذج Hugging Face"""
    if not load_model():
        raise Exception("فشل في تحميل النموذج")
    
    chunks = split_text_into_chunks(text)
    all_questions = []
    
    for chunk in chunks:
        if len(all_questions) >= num_questions:
            break
            
        try:
            if question_type == 'mcq':
                prompt = f"generate mcq question in arabic: {chunk}"
            else:
                prompt = f"generate essay question in arabic: {chunk}"
            
            generated = question_generator(
                prompt,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            question_text = generated[0]['generated_text'].strip()
            
            if question_type == 'mcq':
                parts = question_text.split('\n')
                question = parts[0] if parts else "ما هو الصحيح مما يلي؟"
                choices = parts[1:] if len(parts) > 1 else [chunk, "خيار 2", "خيار 3", "خيار 4"]
                
                # التأكد من وجود 4 خيارات على الأقل
                while len(choices) < 4:
                    choices.append(f"خيار {len(choices) + 1}")
                
                all_questions.append({
                    'question': question,
                    'choices': choices[:4],  # أخذ أول 4 خيارات فقط
                    'correct_answer': 0
                })
            else:
                all_questions.append({
                    'question': question_text,
                    'answer': "يمكنك الإجابة على هذا السؤال بناءً على النص المقدم."
                })
                
        except Exception as e:
            print(f"خطأ في توليد السؤال: {str(e)}")
            continue
    
    # إذا لم نتمكن من توليد أي أسئلة، نستخدم الطريقة البسيطة
    if not all_questions:
        if question_type == 'mcq':
            question = "ما هو الصحيح مما يلي؟"
            choices = [text[:100], "خيار 2", "خيار 3", "خيار 4"]
            all_questions.append({
                'question': question,
                'choices': choices,
                'correct_answer': 0
            })
        else:
            all_questions.append({
                'question': "اشرح النص التالي: " + text[:100],
                'answer': "يمكنك الإجابة على هذا السؤال بناءً على النص المقدم."
            })
    
    return all_questions[:num_questions]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم تحميل أي ملف'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار أي ملف'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'يجب أن يكون الملف بصيغة PDF'}), 400
    
    # التحقق من حجم الملف (16MB كحد أقصى)
    if len(file.read()) > 16 * 1024 * 1024:  # 16MB in bytes
        return jsonify({'error': 'حجم الملف يتجاوز الحد المسموح به (16MB)'}), 400
    
    file.seek(0)  # إعادة مؤشر القراءة إلى بداية الملف
    
    # التحقق من صحة ملف PDF
    if not is_valid_pdf(file):
        return jsonify({'error': 'الملف غير صالح أو تالف'}), 400
    
    file.seek(0)
    
    try:
        # استخراج النص من PDF
        text = extract_text_from_pdf(file)
        if not text.strip():
            return jsonify({'error': 'لم يتم العثور على نص في الملف'}), 400
        
        # تنظيف النص
        text = clean_text(text)
        
        # الحصول على نوع الأسئلة وعددها
        question_type = request.form.get('question_type', 'mcq')
        try:
            num_questions = int(request.form.get('num_questions', 5))
        except ValueError:
            num_questions = 5
        
        # توليد الأسئلة باستخدام النموذج
        questions = generate_questions_from_text(text, num_questions, question_type)
        
        if not questions:
            return jsonify({'error': 'لم نتمكن من توليد أسئلة من النص المقدم'}), 400
        
        return jsonify({'questions': questions})
        
    except Exception as e:
        print(f"خطأ: {str(e)}")
        return jsonify({'error': f'حدث خطأ أثناء معالجة الملف: {str(e)}'}), 500

if __name__ == '__main__':
    # تحميل النموذج عند بدء التطبيق
    load_model()
    app.run(debug=True)
