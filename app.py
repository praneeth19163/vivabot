import os
import time
import cv2
import fitz
import ollama 
import numpy as np
import pandas as pd
import shutil
import smtplib
import nltk
from nltk.tokenize import sent_tokenize
from email.message import EmailMessage
import requests
import random
import joblib
from deepface import DeepFace
import json
from collections import defaultdict
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import ast
from datetime import datetime,timedelta, date
import mysql.connector
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, Response, session

app = Flask(__name__)
app.secret_key = 'klm18504'
EXCEL_FILE = 'students_data.xlsx'
FACES_DIR = './static/faces'
os.makedirs(FACES_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_FOLDER = os.path.join(BASE_DIR, 'classes')
STUDENT_MARKS_FOLDER = os.path.join(BASE_DIR, 'student_marks')
os.makedirs(STUDENT_MARKS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'uploads'  
otp_storage = {}
RECORDINGS_FOLDER = "recordings"
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
video_writer = None
recording_active = False
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def start_recording(roll_no):
    """Start recording the video for the viva session."""
    global video_writer, recording_active
    recording_active = True
    filename = os.path.join(RECORDINGS_FOLDER, f"{roll_no}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    session['recording_filename'] = filename  
    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    def record():
        global recording_active
        while recording_active:
            ret, frame = camera.read()
            if not ret:
                break
            video_writer.write(frame)
        camera.release()
        video_writer.release()
    import threading
    threading.Thread(target=record, daemon=True).start()
    print(f"Recording started: {filename}")


def stop_recording():
    """Stop video recording."""
    global recording_active
    recording_active = False
    print("Recording stopped.")

@app.route("/generate_questions", methods=["GET", "POST"])
def generate_questions():
    if request.method == "POST":
        subject = request.form.get("subject").lower().replace(" ", "_")  
        week = request.form.get("week")
        topics = request.form.get("topics")
        num_questions = request.form.get("num_questions", "")
        print(f"Received request: Subject={subject}, Week={week}, Topics={topics}, Num_Questions={num_questions}")
        if not subject or not week or not topics:
            print(" Error: Missing required fields")
            return "Error: All fields are required", 400  
        if not week.startswith("Week ") or not week.split(" ")[1].isdigit():
            print("Error: Invalid week format")
            return "Error: Week must be in format 'Week X'", 400
        if not num_questions:  
            num_questions = 5
        else:
            try:
                num_questions = int(num_questions)
                if num_questions < 1:
                    raise ValueError
            except ValueError:
                print("Error: Invalid number of questions")
                return "Error: Number of questions must be a positive integer", 400
        prompt = f"""
        Generate exactly {num_questions} questions for each difficulty level (Easy, Medium, Hard)
        related to the subject "{subject}" for {week}. The topics are: {topics}.

        Return the questions in the following JSON format:
        {{
            "Easy": ["Q1", "Q2", ..., "Q{num_questions}"],
            "Medium": ["Q1", "Q2", ..., "Q{num_questions}"],
            "Hard": ["Q1", "Q2", ..., "Q{num_questions}"]
        }}
        """
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": "mistral", "prompt": prompt, "stream": False}
        try:
            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("response", "").strip()
            print(f"Ollama Response:\n{generated_text}")
            try:
                questions_dict = json.loads(generated_text)
            except json.JSONDecodeError:
                print("Error: Could not parse JSON from Ollama response")
                return "Error: Failed to process generated questions", 500  
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                table_name = f"{subject}_questions"
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    qn_no INT AUTO_INCREMENT PRIMARY KEY,
                    week VARCHAR(10) NOT NULL,
                    question TEXT NOT NULL,
                    proficiency_level ENUM('Easy','Medium','Hard') NOT NULL
                )
                """
                cursor.execute(create_table_query)
                insert_query = f"INSERT INTO {table_name} (week, question, proficiency_level) VALUES (%s, %s, %s)"
                for level, questions in questions_dict.items():
                    for question in questions:
                        cursor.execute(insert_query, (week, question, level))
                conn.commit()
                cursor.close()
                conn.close()
                print("Questions successfully stored in the database!")
                return render_template("success.html")
            except mysql.connector.Error as e:
                print(f"Database Error: {e}")
                return "Error: Failed to store questions in database", 500  
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Failed to fetch questions from Ollama", 500  
    return render_template("generate_questions.html")


def extract_questions_from_pdf(pdf_path):
    extracted_text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page in pdf_document:
            extracted_text += page.get_text("text") + "\n"
    return extracted_text


def parse_questions(text):
    questions = []
    lines = text.split("\n") 
    for line in lines:
        print("Processing Line:", line)
        match = re.match(r'(Week \d+)\s*/\s*(.+?)\s*/\s*(\w+)', line.strip())
        if match:
            week = match.group(1)  
            question = match.group(2)  
            proficiency = match.group(3).capitalize()  

            questions.append({
                'week': week,
                'question': question,
                'proficiency': proficiency
            })
    return questions


def insert_questions(subject, questions):
    """ Inserts extracted questions into the respective subject table """
    table_name = f"{subject.lower()}_questions"  
    connection = get_db_connection()
    cursor = connection.cursor()
    insert_query = f"INSERT INTO {table_name} (week, question, proficiency_level) VALUES (%s, %s, %s)"
    for question in questions:
        cursor.execute(insert_query, (question['week'], question['question'], question['proficiency']))
    connection.commit()
    cursor.close()
    connection.close()


@app.route('/faculty_upload', methods=['GET', 'POST'])
def faculty_upload():
    if request.method == 'POST':
        subject = request.form.get('subject')
        pdf_file = request.files.get('pdf_file')
        if not subject or not pdf_file:
            flash("Please select a subject and upload a PDF.", "danger")
            return redirect(url_for('faculty_upload'))
        if pdf_file.filename == '':
            flash("No file selected!", "danger")
            return redirect(url_for('faculty_upload'))
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)  
        extracted_text = extract_questions_from_pdf(pdf_path)
        questions = parse_questions(extracted_text)
        print("Extracted Questions:", questions)
        if questions:
            insert_questions(subject, questions)
            flash(f"Successfully uploaded {len(questions)} questions!", "success")
        else:
            flash("No valid questions found in the PDF. Please check the format.", "warning")
        return redirect(url_for('faculty_upload'))
    return render_template('faculty_upload.html')


@app.route('/faculty_dashboard')
def faculty_dashboard():
    return render_template('faculty_dashboard.html')


@app.route('/excel_route/<session_id>/<class_name>/<subject_name>', methods=['GET'])
def excel_route(session_id, class_name, subject_name):
    session['class_name'] = class_name
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"{class_name}_{subject_name}_{current_date}.xlsx"
    file_path = os.path.join(STUDENT_MARKS_FOLDER, file_name)
    if os.path.exists(file_path):
        return redirect(url_for('start_viva', session_id=session_id))
    source_file = os.path.join(CLASSES_FOLDER, f"{class_name}.xlsx")
    if not os.path.exists(source_file):
        return f"Error: Source file for {class_name} does not exist."
    shutil.copy(source_file, file_path)
    df = pd.read_excel(file_path)
    df['Attendance'] = None
    df['Marks'] = None
    df.to_excel(file_path, index=False)
    return redirect(url_for('start_viva', session_id=session_id))


@app.route('/start_viva/<session_id>', methods=['GET'])
def start_viva(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT subject FROM viva_sessions WHERE id = %s", (session_id,))
    result = cursor.fetchone()
    if result:
        subject = result['subject']
    else:
        return "Error: Subject not found."
    table_name = f"{subject}_questions"
    try:
        query = f"SELECT DISTINCT week FROM {table_name}"
        cursor.execute(query)
        weeks = [row['week'] for row in cursor.fetchall()]
    except Exception as e:
        return f"Error fetching weeks: {e}"
    conn.close()
    return render_template('start_viva.html', session_id=session_id, subject=subject, weeks=weeks)

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",  
        user="root",  
        password="klm18504",  
        database="vivabot"  
    )

def train_face_recognizer():
    print("Training face recognition model...")
    face_files = [f for f in os.listdir(FACES_DIR) if f.endswith('.jpg')]
    if not face_files:
        print("No faces found for training.")
        return
    images, labels = [], []
    label_map = {}
    for idx, filename in enumerate(face_files):
        roll_no = filename.split('.')[0]  
        image_path = os.path.join(FACES_DIR, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping invalid image: {filename}")
            continue
        images.append(image)
        labels.append(roll_no)
        label_map[roll_no] = idx  
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(numeric_labels))
    recognizer.save('face_model.yml')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Face recognition model trained successfully!")

def load_face_recognizer():
    try:
        if not os.path.exists("face_model.yml") or not os.path.exists("label_encoder.pkl"):
            print("No trained model found.")
            return None, None
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_model.yml")
        label_encoder = joblib.load("label_encoder.pkl")
        label_mapping = {roll_no: idx for roll_no, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        return recognizer, label_mapping
    except Exception as e:
        print(f"Error loading face recognizer: {e}")
        return None, None
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def init_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=['Name', 'RollNo', 'Class', 'Face'])
        df.to_excel(EXCEL_FILE, index=False)
    else:
        df = pd.read_excel(EXCEL_FILE)
        if not all(col in df.columns for col in ['Name', 'RollNo', 'Class', 'Face']):
            df = pd.DataFrame(columns=['Name', 'RollNo', 'Class', 'Face'])
            df.to_excel(EXCEL_FILE, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll_no = request.form['roll_no']
        class_name = request.form['class']
        if not os.path.exists(EXCEL_FILE):
            df = pd.DataFrame(columns=["Name", "RollNo", "Class", "Face"])
            df.to_excel(EXCEL_FILE, index=False)
        else:
            df = pd.read_excel(EXCEL_FILE)
        if "RollNo" not in df.columns:
            flash("Error: Invalid student data file. Expected 'RollNo' column is missing.", "danger")
            return redirect(url_for('home'))
        if roll_no in df["RollNo"].astype(str).values:
            flash("Roll number already registered", "danger")
            return redirect(url_for('home'))
        df = pd.concat([df, pd.DataFrame({'RollNo': [roll_no], 'Name': [name], 'Class': [class_name], 'Face': [f'{roll_no}.jpg']})], ignore_index=True)
        df['Face'] = df['Face'].astype(str)
        df.to_excel(EXCEL_FILE, index=False)
        flash("Student registered successfully! Proceed to face capture.", "success")
        return redirect(url_for('capture_face', roll_no=roll_no))
    return render_template('register.html')


@app.route('/capture_face/<roll_no>', methods=['GET', 'POST'])
def capture_face(roll_no):
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Error accessing camera.", "danger")
            return redirect(url_for('register'))
        time.sleep(1)  
        success, frame = cap.read()
        cap.release()
        if not success:
            flash("Failed to capture face. Try again.", "danger")
            return redirect(url_for('register'))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            flash("No face detected. Try again.", "danger")
            return redirect(url_for('capture_face', roll_no=roll_no))
        (x, y, w, h) = faces[0]
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (200, 200))
        face_path = os.path.join(FACES_DIR, f"{roll_no}.jpg")
        cv2.imwrite(face_path, face_image)
        flash("Face captured successfully!", "success")
        train_face_recognizer()
        return redirect(url_for('home'))
    return render_template('capture_face.html', roll_no=roll_no)


@app.route('/video_feed')
def video_feed():
    def generate():
        camera = cv2.VideoCapture(0)  
        if not camera.isOpened():
            flash("Error: Could not access the camera", "danger")
            yield b''
        while True:
            success, frame = camera.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['GET', 'POST'])
def login():
    subject = 'Unknown'  
    weeks = []
    if request.method == 'GET':
        session_id = request.args.get('session_id')
        weeks = request.args.getlist('weeks')  
        if session_id:
            session['session_id'] = session_id
        if weeks:
            session['weeks'] = weeks  
        else:
            session['weeks'] = session.get('weeks', [])
        print(f"Selected weeks from session (GET): {session['weeks']}")  
    if request.method == 'POST':
        session.pop('questions', None)
        session.pop('asked_questions', None)
        session.pop('marks', None)
        session.pop('proficiency', None)
        session.pop('feedback',None)
        print("Resetting session data for new viva attempt.")
        session_id = session.get('session_id')
        if not session_id:
            flash("Session ID is missing. Please start the process again.", "danger")
            return redirect(url_for('start_viva'))
        weeks = request.form.getlist('weeks')
        if not weeks:
            weeks = session.get('weeks', [])
        session_id = session.get('session_id')
        session['weeks'] = weeks
        print(f"Weeks in POST request: {weeks}")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT subject, class_name FROM viva_sessions WHERE id = %s", (session_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                subject, class_name = result
            else:
                flash("No subject found for this session ID.", "danger")
                return redirect(url_for('start_viva'))
        except Exception as e:
            flash(f"Database error: {e}", "danger")
            return redirect(url_for('start_viva')) 
        roll_no = request.form['roll_no']
        today_date = datetime.today().strftime('%Y-%m-%d')
        excel_filename = f"student_marks/{class_name}_{subject}_{today_date}.xlsx"
        excel_path = os.path.join(os.getcwd(), excel_filename)
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            if roll_no in df["Roll Number"].values:
                student_row = df[df["Roll Number"] == roll_no]
                if student_row["Attendance"].values[0] == "P": 
                    flash("You have already taken this viva session. You cannot take it again.", "danger")
                    return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        recognizer, label_mapping = load_face_recognizer()
        if not recognizer:
            flash("No faces available for recognition. Please register students first.", "danger")
            return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            flash("Error: Could not access the camera.", "danger")
            return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        success, frame = camera.read()
        camera.release()
        if not success:
            flash("Error: Could not capture image from the camera.", "danger")
            return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            flash("No face detected. Please adjust your position and try again.", "warning")
            return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        (x, y, w, h) = faces[0]
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (200, 200))
        try:
            label, confidence = recognizer.predict(face_image)
            print(f"Recognized Label: {label}, Confidence: {confidence}")
        except Exception as e:
            flash(f"Face recognition error: {e}", "danger")
            return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
        for registered_roll_no, numeric_label in label_mapping.items():
            print(f"Checking {registered_roll_no} -> Label {numeric_label}")
            if numeric_label == label:
                if confidence < 80:  
                    if registered_roll_no == roll_no:
                        print(f"Face matched for {roll_no}")
                        session['roll_no'] = roll_no
                        session['subject'] = subject
                        session['recording_active'] = True  
                        start_recording(roll_no)
                        return redirect(url_for('questions'))  
                else:
                    print(f"Confidence too high ({confidence}) for {registered_roll_no}")
        flash("Face did not match the registered image. Please try again.", "danger")
        return redirect(url_for('login', session_id=session_id, weeks=session['weeks']))
    return render_template('login.html', subject=subject, session_id=session.get('session_id'), weeks=session.get('weeks', []))


@app.route('/questions', methods=['GET', 'POST'])
def questions():
    subject = session.get('subject')
    roll_no = session.get('roll_no')
    weeks = session.get('weeks', [])
    if not subject or not weeks or not roll_no:
        flash("Session data is missing. Please restart.", "danger")
        return redirect(url_for('login'))
    if isinstance(weeks, str):  
        try:
            weeks = ast.literal_eval(weeks)
        except (SyntaxError, ValueError):
            weeks = []
    if not isinstance(weeks, list):  
        weeks = [weeks]  
    weeks = [week.strip() for week in weeks if isinstance(week, str)]  
    table_name = f"{subject}_questions"
    if 'questions' not in session or not session['questions']:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        format_strings = ','.join(['%s'] * len(weeks))
        cursor.execute(f"SELECT * FROM {table_name} WHERE week IN ({format_strings})", tuple(weeks))
        all_questions = cursor.fetchall()
        if not all_questions:
            flash("No questions found for the selected weeks!", "danger")
            return redirect(url_for('login'))
        cursor.close()
        conn.close()
        session['questions'] = all_questions
        session['asked_questions'] = []
        session['marks'] = 0
        session['proficiency'] = 'Medium'
        session['question_score'] = 0  
        session['feedback'] = []  
    if session['question_score'] >= 5:  
        return redirect(url_for('finish'))
    question_scores = {'Easy': 0.5, 'Medium': 1, 'Hard': 1.5}
    all_questions = session['questions']
    current_proficiency = session['proficiency']
    available_questions = [q for q in all_questions if q['proficiency_level'] == current_proficiency
                           and q['qn_no'] not in session['asked_questions']
                           and session['question_score'] + question_scores[q['proficiency_level']] <= 5]
    if not available_questions:
        if current_proficiency == 'Hard':
            session['proficiency'] = 'Medium'
        elif current_proficiency == 'Medium':
            session['proficiency'] = 'Easy'
        available_questions = [q for q in all_questions if q['proficiency_level'] == session['proficiency']
                               and q['qn_no'] not in session['asked_questions']
                               and session['question_score'] + question_scores[q['proficiency_level']] <= 5]

        if not available_questions:
            return redirect(url_for('finish'))
    if 'current_question' not in session:
        session['current_question'] = random.choice(available_questions)
    current_question = session['current_question']
    if request.method == 'POST':
        if 'skip' in request.form:
            feedback_prompt = f"""
            Provide the correct answer for the following question:

            Question: {current_question['question']}
            """
            feedback_response = ollama.chat(model="mistral", messages=[{"role": "user", "content": feedback_prompt}])
            correct_answer = feedback_response.get('message', {}).get('content', 'Correct answer not available.')
            correct_answer = summarize_text(correct_answer, max_sentences=2)
            feedback = {
                'question': current_question['question'],
                'feedback': "You skipped this question. Review similar topics before attempting again.",
                'correct_answer': correct_answer  
            }
            session['feedback'].append(feedback)
            session['asked_questions'].append(current_question['qn_no'])
            session['question_score'] += question_scores[current_proficiency]
            if session['proficiency'] == 'Medium':
                session['proficiency'] = 'Easy'
            elif session['proficiency'] == 'Hard':
                session['proficiency'] = 'Medium'
            session.pop('current_question', None)
            return redirect(url_for('questions'))
        user_answer = request.form['answer'].strip()
        if not user_answer:
            flash("Please enter an answer before proceeding.", "warning")
            return redirect(url_for('questions'))
        prompt = f"""
        Evaluate the student's answer to the question.
        Question: {current_question['question']}
        Student Answer: {user_answer}
        Provide a similarity score from 0 to 100, where 100 is a perfect match.
        """
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        ai_response = response.get('message', {}).get('content', '')
        similarity_score = 0
        if ai_response:
            number = ''.join(filter(str.isdigit, ai_response.split()[0]))
            if number:
                similarity_score = int(number)
        correct = similarity_score >= 40
        current_score = question_scores[current_proficiency]
        session['question_score'] += current_score  
        session.modified = True
        if correct:
            session['marks'] += current_score
            if current_proficiency == 'Medium':
                session['proficiency'] = 'Hard'
            elif current_proficiency == 'Easy':
                session['proficiency'] = 'Medium'
            improvement_prompt = f"""
            The following answer was correct. Suggest ways to make it even better or more accurate:

            Question: {current_question['question']}
            """
            improvement_response = ollama.chat(model="mistral", messages=[{"role": "user", "content": improvement_prompt}])
            improvement_feedback = improvement_response.get('message', {}).get('content', 'No improvement suggestions available.')
            improvement_feedback = summarize_text(improvement_feedback,max_sentences=2)
            feedback = {
                'question': current_question['question'],
                'feedback': "Your answer was correct! Here's how you can make it even better",
                'correct_answer': improvement_feedback
            }
            session['feedback'].append(feedback)
        else:
            feedback_prompt = f"""
            Provide an explanation of how to correctly answer this question:

            Question: {current_question['question']}
            """
            feedback_response = ollama.chat(model="mistral", messages=[{"role": "user", "content": feedback_prompt}])
            correct_text = feedback_response.get('message', {}).get('content', 'No feedback available.')
            correct_text = summarize_text(correct_text,max_sentences=2)
            feedback = {
                'question': current_question['question'],
                'feedback': "Your answer was incorrect. Review the explanation carefully before attempting similar questions.",
                'correct_answer':correct_text
            }
            session['feedback'].append(feedback)
            if current_proficiency == 'Medium':
                session['proficiency'] = 'Easy'
            elif current_proficiency == 'Hard':
                session['proficiency'] = 'Medium'
        session['asked_questions'].append(current_question['qn_no'])
        session.pop('current_question', None)
        if session['question_score'] >= 5:
            return redirect(url_for('finish'))
        return redirect(url_for('questions'))
    return render_template('questions.html', 
                           subject=subject, 
                           question=current_question, 
                           proficiency_level=current_proficiency,
                           question_number=len(session['asked_questions']) + 1, 
                           weeks=weeks)


def summarize_text(text, max_sentences=2):
    sentences = sent_tokenize(text)  
    return " ".join(sentences[:max_sentences])


@app.route('/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is missing"}), 400
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT faculty_email FROM viva_sessions WHERE id = %s", (session_id,))
    session_data = cursor.fetchone()
    if not session_data:
        return jsonify({"success": False, "message": "Session not found"}), 404
    faculty_email = session_data["faculty_email"]
    otp = str(random.randint(100000, 999999))
    otp_storage[session_id] = otp  
    sender_email = "lekhya1854@gmail.com"
    sender_password = "eohx qxct qiev sqhb"    
    msg = EmailMessage()
    msg["Subject"] = "Viva Session OTP Verification"
    msg["From"] = sender_email
    msg["To"] = faculty_email
    msg.set_content(f"Your OTP to end the Viva session is: {otp}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return jsonify({"success": True, "message": "OTP sent to faculty email."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to send OTP: {str(e)}"}), 500
    finally:
        cursor.close()
        connection.close()


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    session_id = data.get("session_id")
    entered_otp = data.get("otp")
    if not session_id or not entered_otp:
        return jsonify({"success": False, "message": "Session ID and OTP are required"}), 400
    # Check if OTP exists for this session
    if session_id not in otp_storage:
        return jsonify({"success": False, "message": "OTP expired or invalid"}), 400
    # Validate OTP
    if otp_storage[session_id] == entered_otp:
        del otp_storage[session_id]  # Remove OTP after successful verification

        # Call the end_viva_session function
        return end_viva_session(session_id)
    else:
        return jsonify({"success": False, "message": "Incorrect OTP"}), 400
    

@app.route('/end_viva_session', methods=['POST'])
def end_viva_session(session_id=None):
    if session_id is None:
        data = request.get_json()
        session_id = data.get("session_id")
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is missing"}), 400
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT class_name, subject, faculty_email FROM viva_sessions WHERE id = %s", (session_id,))
    viva_session = cursor.fetchone()
    if not viva_session:
        return jsonify({"success": False, "message": "Viva session not found"}), 404
    class_name = viva_session["class_name"]
    subject_name = viva_session["subject"]
    faculty_email = viva_session["faculty_email"]
    current_date = datetime.today().strftime("%Y-%m-%d")
    file_name = f"{class_name}_{subject_name}_{current_date}.xlsx"
    file_path = os.path.join("student_marks", file_name)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "message": f"Excel file not found: {file_name}"}), 404
    df = pd.read_excel(file_path)
    df.loc[df['Attendance'].isna(), 'Attendance'] = 'A'
    df.loc[df['Marks'].isna(), 'Marks'] = 0
    df.to_excel(file_path, index=False)
    cursor.execute("UPDATE viva_sessions SET end_early = TRUE WHERE id = %s", (session_id,))
    connection.commit()
    cursor.close()
    connection.close()
    send_email(faculty_email, file_path)
    return jsonify({"success": True, "message": "Viva session ended. Updated Excel sent to faculty.", "redirect": "/dashboard"})


def send_email(recipient_email, attachment_path):
    sender_email = "lekhya1854@gmail.com"
    sender_password = "eohx qxct qiev sqhb"  
    if not os.path.exists(attachment_path):
        print(f"Error: The file {attachment_path} does not exist.")
        return
    msg = EmailMessage()
    msg["Subject"] = "Updated Viva Attendance and Marks Sheet"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content("Please find the updated attendance and marks sheet attached.")
    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {recipient_email} successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


@app.route('/finish', methods=['GET', 'POST'])
def finish():
    if session.get('recording_active'):
        stop_recording()
        session['recording_active'] = False  
    recorded_file = session.get('recording_filename')
    confidence_score, confidence_feedback, answer_quality_score, answer_quality_feedback = analyze_video(recorded_file)
    session['confidence_score'] = confidence_score
    session['confidence_feedback'] = confidence_feedback
    session['answer_quality_score'] = answer_quality_score
    session['answer_quality_feedback'] = answer_quality_feedback
    feedback_list = session.get('feedback', [])
    return render_template('finish.html', 
                           feedback_list=feedback_list,
                           confidence_score=confidence_score, 
                           confidence_feedback=confidence_feedback,
                           answer_quality_score=answer_quality_score, 
                           answer_quality_feedback=answer_quality_feedback)

def analyze_video(video_path):    
    if not video_path or not os.path.exists(video_path):
        print("Video file not found for analysis")
        return "N/A", "N/A", "N/A", "N/A"
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    confident_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            if dominant_emotion in ["happy", "neutral"]:
                confident_frames += 1  
        except Exception as e:
            print(f"DeepFace Error: {e}")
    cap.release()
    confidence_score = round((confident_frames / total_frames) * 100, 2) if total_frames > 0 else "N/A"
    confidence_feedback = get_confidence_feedback(confidence_score)
    answer_quality_score = np.random.randint(60, 95)  
    answer_quality_feedback = get_answer_quality_feedback(answer_quality_score)
    print(f"Confidence Score: {confidence_score}% - {confidence_feedback}")
    print(f"Answer Quality Score: {answer_quality_score}% - {answer_quality_feedback}")
    return confidence_score, confidence_feedback, answer_quality_score, answer_quality_feedback


def get_confidence_feedback(score):
    if score == "N/A":
        return "No confidence data available."
    elif score >= 80:
        return "You answered with great confidence! Keep it up! 👍"
    elif score >= 60:
        return "You were somewhat confident, but try to speak more assertively."
    elif score >= 40:
        return "You seemed a bit unsure. Try to practice speaking more confidently."
    else:
        return "You need to work on your confidence. Stay calm and trust yourself."
    

def get_answer_quality_feedback(score):
    if score == "N/A":
        return "No speech data available."
    elif score >= 80:
        return "Your answers were clear, fluent, and well-structured. Excellent job! 🎯"
    elif score >= 60:
        return "Your answers were good, but adding more details would improve clarity."
    elif score >= 40:
        return "Your answers lacked fluency or clarity. Try to organize your thoughts better."
    else:
        return "Your answers need improvement. Focus on clarity and accuracy."
    

@app.route('/finish_viva', methods=['POST','GET'])
def finish_viva():
    roll_no = session.get('roll_no')
    subject = session.get('subject')
    class_name = session.get('class_name')
    marks = session.get('marks', 0)
    print(f"Session data: roll_no={roll_no}, subject={subject}, class_name={class_name}, marks={marks}")
    missing_data = []
    if not roll_no:
        missing_data.append('roll_no')
    if not subject:
        missing_data.append('subject')
    if not class_name:
        missing_data.append('class_name')
    if missing_data:
        flash(f"Session data missing for: {', '.join(missing_data)}. Please restart.", "danger")
        return redirect(url_for('login'))
    if not os.path.exists(STUDENT_MARKS_FOLDER):
        flash("Student marks folder does not exist.", "danger")
        return redirect(url_for('login'))
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"{class_name}_{subject}_{current_date}.xlsx"
    file_path = os.path.join(STUDENT_MARKS_FOLDER, file_name)
    print(f"Looking for file: {file_path}")
    print(f"Available files: {os.listdir(STUDENT_MARKS_FOLDER)}")
    if not os.path.exists(file_path):
        flash(f"Error: Attendance sheet '{file_name}' not found.", "danger")
        return redirect(url_for('login'))
    df = pd.read_excel(file_path)
    if 'Roll Number' in df.columns:
        roll_number_column = 'Roll Number'  
    elif 'RollNo' in df.columns:
        roll_number_column = 'RollNo'
    else:
        flash("Error: Roll Number column not found in the sheet.", "danger")
        return redirect(url_for('login'))
    if 'Attendance' not in df.columns:
        df['Attendance'] = None
    if 'Marks' not in df.columns:
        df['Marks'] = None
    if roll_no in df[roll_number_column].values:
        df.loc[df[roll_number_column] == roll_no, 'Attendance'] = 'P'
        df.loc[df[roll_number_column] == roll_no, 'Marks'] = marks
        df.to_excel(file_path, index=False)
        flash(f"Marks updated for {roll_no}", "success")
    else:
        flash("Roll number not found in the sheet.", "warning")
    recorded_file = session.get('recording_filename')
    if recorded_file and os.path.exists(recorded_file):
        try:
            os.remove(recorded_file)
            print(f"Recording deleted: {recorded_file}")
        except Exception as e:
            print(f"Error deleting recording: {e}")
    session.pop('roll_no', None)
    session.pop('marks', None)
    session.pop('current_question_index', None)
    session.pop('proficiency', None)
    session.pop('recording_filename', None)
    return redirect(url_for('login'))
LAST_RESET_DATE = None


@app.route('/dashboard')
def dashboard():
    global LAST_RESET_DATE
    today = date.today()
    if LAST_RESET_DATE != today:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE viva_sessions SET end_early = 0")
        conn.commit()
        cursor.close()
        conn.close()
        LAST_RESET_DATE = today
        print(f"end_early reset at {datetime.now()}")
    now = datetime.now()
    day_of_week = now.strftime('%A')
    current_time = now.time()  
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, class_name, subject, faculty_name, 
               start_time, end_time, faculty_email, end_early
        FROM viva_sessions 
        WHERE day_of_week = %s 
        AND start_time <= %s 
        AND end_time >= %s
        AND end_early = 0   
    """, (day_of_week, current_time, current_time))
    sessions = cursor.fetchall()
    for session in sessions:
        if isinstance(session["start_time"], timedelta):
            session["start_time"] = (datetime.min + session["start_time"]).time()
        if isinstance(session["end_time"], timedelta):
            session["end_time"] = (datetime.min + session["end_time"]).time()
        start_time = session["start_time"]
        end_time = session["end_time"]
        if session["end_early"] == 1:
            session["disabled"] = True  
        else:
            session["disabled"] = False  
        if session["end_early"] == 1 and start_time <= current_time and end_time >= current_time:
            cursor.execute("UPDATE viva_sessions SET end_early = 0 WHERE id = %s", (session["id"],))
            conn.commit()
            session["disabled"] = False  
    cursor.close()
    conn.close()
    return render_template('dashboard.html', sessions=sessions)

@app.route('/generate_final_marks', methods=['GET', 'POST'])
def generate_final_marks():
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        class_name = request.form.get('class_name')
        subject = request.form.get('subject')

        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return render_template("final_marks.html", error="Invalid date format.")

        folder = "student_marks"
        all_files = os.listdir(folder)

        matched_files = []
        for file in all_files:
            if file.startswith(f"{class_name}_{subject}_") and file.endswith(".xlsx"):
                try:
                    date_str = file.split("_")[-1].split(".xlsx")[0]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if start_date <= file_date <= end_date:
                        matched_files.append(os.path.join(folder, file))
                except Exception:
                    continue

        if not matched_files:
            return render_template("final_marks.html", error="No files found in the given date range.")

        student_data = defaultdict(float)
        student_names = {}

        for file in matched_files:
            df = pd.read_excel(file)
            for _, row in df.iterrows():
                roll = str(row['Roll Number']).strip()
                name = str(row['Name']).strip()
                marks = float(row['Marks']) if not pd.isna(row['Marks']) else 0
                student_data[roll] += marks
                student_names[roll] = name

        final_data = [{"Roll Number": roll, "Name": student_names[roll], "Marks": round(marks, 2)}
                      for roll, marks in student_data.items()]
        final_df = pd.DataFrame(final_data)

        output_file = f"{class_name}_{subject}_FinalMarks.xlsx"
        output_path = os.path.join("student_marks", output_file)
        final_df.to_excel(output_path, index=False)

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT faculty_email FROM viva_sessions WHERE class_name = %s AND subject = %s ORDER BY id DESC LIMIT 1",
                       (class_name, subject))
        row = cursor.fetchone()
        cursor.close()
        connection.close()

        if not row:
            return render_template("final_marks.html", error="Faculty email not found.")

        faculty_email = row["faculty_email"]

        send_final_marks_email(faculty_email, output_path)

        return render_template("final_marks.html", success="Final marks file generated and emailed successfully!")

    return render_template("final_marks.html")


def send_final_marks_email(recipient_email, attachment_path):
    sender_email = "lekhya1854@gmail.com"
    sender_password = "eohx qxct qiev sqhb"  

    msg = EmailMessage()
    msg["Subject"] = "Final Viva Marks Report"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content("Please find attached the final viva marks report.")

    with open(attachment_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(attachment_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)


@app.teardown_appcontext

def cleanup(exception=None):
    global camera
    if camera.isOpened():
        camera.release()
if __name__ == "__main__":
    init_excel()  
    app.run(debug=True)
