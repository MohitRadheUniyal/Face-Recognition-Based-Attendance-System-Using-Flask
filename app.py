import cv2
import os
import shutil
from flask import Flask, request, render_template, send_file, Response, session, redirect, url_for, flash
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Subject,Type')

# Predefined lists
BRANCHES = [
    'Computer Science', 'Mechanical Engineering', 'Electrical Engineering',
    'Civil Engineering', 'Electronics Engineering'
]
SUBJECTS = [
    'Mathematics', 'Physics', 'Chemistry', 'Computer Science', 'Engineering Drawing'
]

# User Model for SQLite
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.String(20), unique=True, nullable=False)
    branch = db.Column(db.String(50), nullable=True)  # For students
    subjects = db.Column(db.String(200), nullable=True)  # For teachers

# Create database tables
with app.app_context():
    db.create_all()

# Utility Functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance(selected_date=datetoday, subject='All'):
    try:
        df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
        if 'Subject' not in df.columns:
            df['Subject'] = 'Unknown'  # Default value if Subject is missing
        if subject != 'All':
            df = df[df['Subject'] == subject]
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        subjects = df['Subject']
        types = df['Type'] if 'Type' in df.columns else ['Regular'] * len(df)  # Default type if missing
        l = len(df)
        attendance = [(n, r, t, s, ty) for n, r, t, s, ty in zip(names, rolls, times, subjects, types)]
    except FileNotFoundError:
        names, rolls, times, subjects, types, l, attendance = [], [], [], [], [], 0, []
    return names, rolls, times, l, attendance

def add_attendance(name, selected_date=datetoday, subject='Mathematics', att_type='Regular'):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    if f'Attendance-{selected_date}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{selected_date}.csv', 'w') as f:
            f.write('Name,Roll,Time,Subject,Type')
    df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
    if int(userid) not in list(df[(df['Subject'] == subject) & (df['Type'] == att_type)]['Roll']):
        with open(f'Attendance/Attendance-{selected_date}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{subject},{att_type}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, l

def deletefolder(duser):
    if os.path.exists(duser):
        pics = os.listdir(duser)
        for i in pics:
            os.remove(duser+'/'+i)
        os.rmdir(duser)

def get_attendance_dates():
    files = os.listdir('Attendance')
    dates = [f.split('-')[1].split('.')[0] for f in files if f.startswith('Attendance-')]
    return sorted(dates, reverse=True)

# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].lower().replace(' ', '_')
        password = request.form['password']
        name = request.form['name']
        user_id = request.form['user_id']
        role = request.form['role']
        
        # Handle multiple subject selection for teachers
        subjects = ''
        if role == 'teacher':
            subjects = request.form.get('subjects', '').strip()
            if not subjects:
                flash('Please enter a subject for teacher role.', 'danger')
                return redirect(url_for('signup'))

        # Restrict signup to admin and teacher roles
        if role not in ['admin', 'teacher']:
            flash('Invalid role selected. Only admins and teachers can sign up here.', 'danger')
            return redirect(url_for('signup'))

        # Check for existing username or user_id
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

        if User.query.filter_by(user_id=user_id).first():
            flash('User ID already exists!', 'danger')
            return redirect(url_for('signup'))

        # Add user to database
        new_user = User(
            username=username,
            password=generate_password_hash(password),
            role=role,
            name=name,
            user_id=user_id,
            subjects=subjects
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', subjects=SUBJECTS)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            # For students, check if password matches hashed user_id
            if user.role == 'student' and check_password_hash(user.password, user.user_id):
                if password == user.user_id:
                    session['username'] = username
                    session['role'] = user.role
                    flash('Login successful!', 'success')
                    return redirect(url_for('student_dashboard'))
            # For admins/teachers, check provided password
            elif user.role in ['admin', 'teacher'] and check_password_hash(user.password, password):
                session['username'] = username
                session['role'] = user.role
                flash('Login successful!', 'success')
                if user.role == 'admin':
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('teacher_dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# Dashboard Routes
@app.route('/student')
def student_dashboard():
    if 'username' not in session or session['role'] != 'student':
        flash('Please log in as a student.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    subject = request.args.get('subject', 'All')
    names, rolls, times, l, attendance = extract_attendance(selected_date, subject)
    user_attendance = [(n, t, s, ty) for n, r, t, s, ty in attendance if str(r) == user.user_id]
    return render_template('student_dashboard.html', name=user.name, id=user.user_id, branch=user.branch,
                           attendance=user_attendance, datetoday2=datetoday2,
                           dates=get_attendance_dates(), selected_date=selected_date,
                           subjects=SUBJECTS, selected_subject=subject)

@app.route('/teacher')
def teacher_dashboard():
    if 'username' not in session or session['role'] != 'teacher':
        flash('Please log in as a teacher.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))

    # Get teacher's subjects
    teacher_subjects = []
    if user.subjects:
        teacher_subjects = [s.strip() for s in user.subjects.split(',')]
    if not teacher_subjects:
        flash('No subjects assigned to you. Please contact admin.', 'warning')
        return redirect(url_for('logout'))

    selected_date = request.args.get('selected_date', datetoday)
    # Set default subject to first assigned subject
    subject = request.args.get('subject', teacher_subjects[0])
    
    # Ensure selected subject is one of teacher's subjects
    if subject not in teacher_subjects:
        subject = teacher_subjects[0]
        flash('Invalid subject selected. Showing attendance for your first assigned subject.', 'warning')
    
    names, rolls, times, l, attendance = extract_attendance(selected_date, subject)
    # Filter attendance to only show selected subject
    attendance = [a for a in attendance if a[3] == subject]

    return render_template('teacher_dashboard.html', 
                         name=user.name, 
                         id=user.user_id,
                         attendance=attendance, 
                         l=len(attendance), 
                         totalreg=totalreg(),
                         datetoday2=datetoday2, 
                         dates=get_attendance_dates(), 
                         selected_date=selected_date, 
                         subjects=teacher_subjects, 
                         selected_subject=subject)

@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    subject = request.args.get('subject', 'All')
    names, rolls, times, l, attendance = extract_attendance(selected_date, subject)
    students = User.query.filter_by(role='student').all()
    teachers = User.query.filter_by(role='teacher').all()
    # Get all subjects from teachers
    all_subjects = set()
    for teacher in teachers:
        if teacher.subjects:
            all_subjects.update(teacher.subjects.split(','))
    return render_template('admin_dashboard.html', name=user.name, id=user.user_id,
                           attendance=attendance, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, dates=get_attendance_dates(), 
                           selected_date=selected_date, subjects=list(all_subjects), 
                           selected_subject=subject, students=students, teachers=teachers)

# Additional Routes
@app.route('/')
def home():
    if 'username' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif session['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/listusers')
def listusers():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    students = User.query.filter_by(role='student').all()
    return render_template('listusers.html', students=students, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))

    duser = request.args.get('user')
    if not duser:
        flash('No user specified for deletion.', 'danger')
        return redirect(url_for('listusers'))

    try:
        username, user_id = duser.split('_')
    except ValueError:
        flash('Invalid user format.', 'danger')
        return redirect(url_for('listusers'))

    user_to_delete = User.query.filter_by(username=username, user_id=user_id).first()
    if user_to_delete:
        db.session.delete(user_to_delete)
        db.session.commit()
    else:
        flash('User not found in database.', 'warning')

    user_folder = f'static/faces/{duser}'
    if os.path.exists(user_folder):
        try:
            deletefolder(user_folder)
        except Exception as e:
            flash(f'Error deleting face images: {str(e)}', 'danger')
            return redirect(url_for('listusers'))

    if os.listdir('static/faces'):
        try:
            train_model()
        except Exception as e:
            flash(f'Error retraining model: {str(e)}', 'warning')
    else:
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')

    flash('User deleted successfully!', 'success')
    return redirect(url_for('listusers'))

@app.route('/download', methods=['GET'])
def download():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))

    selected_date = request.args.get('selected_date', datetoday)
    subject = request.args.get('subject', 'All')

    # Check if the subject is assigned to the teacher
    if user.role == 'teacher':
        if not user.subjects:
            flash('No subjects assigned to you. Please contact admin.', 'danger')
            return redirect(url_for('teacher_dashboard'))
        
        teacher_subjects = [s.strip() for s in user.subjects.split(',')]
        if subject != 'All' and subject not in teacher_subjects:
            flash('You are not authorized to download attendance for this subject.', 'danger')
            return redirect(url_for('teacher_dashboard'))

    file_path = f'Attendance/Attendance-{selected_date}.csv'
    try:
        df = pd.read_csv(file_path)
        if subject != 'All':
            df = df[df['Subject'] == subject]
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f'Attendance-{selected_date}.csv'
        )
    except FileNotFoundError:
        flash('No attendance data for selected date.', 'danger')
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/download_excel', methods=['GET'])
def download_excel():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))

    selected_date = request.args.get('selected_date', datetoday)
    subject = request.args.get('subject', 'All')

    # Check if the subject is assigned to the teacher
    if user.role == 'teacher':
        if not user.subjects:
            flash('No subjects assigned to you. Please contact admin.', 'danger')
            return redirect(url_for('teacher_dashboard'))
        
        teacher_subjects = [s.strip() for s in user.subjects.split(',')]
        if subject != 'All' and subject not in teacher_subjects:
            flash('You are not authorized to download attendance for this subject.', 'danger')
            return redirect(url_for('teacher_dashboard'))

    file_path = f'Attendance/Attendance-{selected_date}.csv'
    try:
        df = pd.read_csv(file_path)
        if subject != 'All':
            df = df[df['Subject'] == subject]
        excel_path = f'Attendance/Attendance-{selected_date}.xlsx'
        df.to_excel(excel_path, index=False)
        return send_file(
            excel_path,
            as_attachment=True,
            download_name=f'Attendance-{selected_date}.xlsx'
        )
    except FileNotFoundError:
        flash('No attendance data for selected date.', 'danger')
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/start', methods=['GET'])
def start():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))

    selected_date = request.args.get('selected_date', datetoday)
    subject = request.args.get('subject', '')
    att_type = request.args.get('type', 'Regular')
    
    if att_type not in ['Regular', 'Evening']:
        att_type = 'Regular'

    # Check if the subject is assigned to the teacher
    if user.role == 'teacher':
        if not user.subjects:
            flash('No subjects assigned to you. Please contact admin.', 'danger')
            return redirect(url_for('teacher_dashboard'))
        
        teacher_subjects = [s.strip() for s in user.subjects.split(',')]
        if not subject or subject not in teacher_subjects:
            flash('You are not authorized to mark attendance for this subject.', 'danger')
            return redirect(url_for('teacher_dashboard'))

    # Check if face recognition model exists
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash('No trained model found. Please add a new face to continue.', 'danger')
        return redirect(url_for('admin_dashboard'))

    names, rolls, times, l, attendance = extract_attendance(selected_date, subject)
    
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person, selected_date, subject, att_type)
            cv2.putText(frame, f'{identified_person} ({subject})', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    flash('Attendance recorded successfully!', 'success')
    return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        branch = request.form.get('branch', '')

        # Validate user_id (11 digits)
        if not (newuserid.isdigit() and len(newuserid) == 11):
            flash('User ID must be exactly 11 digits.', 'danger')
            return redirect(url_for('admin_dashboard'))

        # Set username to user_id for students
        normalized_username = newuserid

        # Check if user exists
        existing_user = User.query.filter_by(user_id=newuserid).first()

        if existing_user:
            # User exists, update face data
            userimagefolder = f'static/faces/{existing_user.name}_{newuserid}'
            if os.path.exists(userimagefolder):
                shutil.rmtree(userimagefolder)
        else:
            # Create new user
            if User.query.filter_by(username=normalized_username).first():
                flash('User ID already exists as a username!', 'danger')
                return redirect(url_for('admin_dashboard'))
            new_user = User(
                username=normalized_username,
                password=generate_password_hash(newuserid),
                role='student',
                name=newusername,
                user_id=newuserid,
                branch=branch
            )
            db.session.add(new_user)
            db.session.commit()
            userimagefolder = f'static/faces/{newusername}_{newuserid}'

        # Capture face images
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access webcam!', 'danger')
            if not existing_user:
                db.session.delete(new_user)
                db.session.commit()
            return redirect(url_for('admin_dashboard'))
        while True:
            ret, frame = cap.read()
            if not ret:
                flash('Error capturing video!', 'danger')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                cap.release()
                return redirect(url_for('admin_dashboard'))
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                flash('Face capture cancelled.', 'warning')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                deletefolder(userimagefolder)
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('admin_dashboard'))
        cap.release()
        cv2.destroyAllWindows()

        # Train the model
        try:
            train_model()
            flash('Student added and face data captured successfully!', 'success')
        except Exception as e:
            flash(f'Error training model: {str(e)}', 'danger')
            if not existing_user:
                db.session.delete(new_user)
                db.session.commit()
            deletefolder(userimagefolder)
            return redirect(url_for('admin_dashboard'))

    return redirect(url_for('admin_dashboard'))

@app.route('/edit_student', methods=['GET', 'POST'])
def edit_student():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    
    user_id = request.args.get('user_id')
    student = User.query.filter_by(user_id=user_id, role='student').first()
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        new_name = request.form['name']
        new_user_id = request.form['user_id']
        branch = request.form.get('branch', '')

        # Validate user_id (11 digits)
        if not (new_user_id.isdigit() and len(new_user_id) == 11):
            flash('User ID must be exactly 11 digits.', 'danger')
            return redirect(url_for('edit_student', user_id=user_id))

        # Check for user_id conflicts
        existing_user = User.query.filter_by(user_id=new_user_id).first()
        if existing_user and existing_user.id != student.id:
            flash('User ID already exists!', 'danger')
            return redirect(url_for('edit_student', user_id=user_id))

        # Update face folder if user_id or name changes
        old_folder = f'static/faces/{student.name}_{student.user_id}'
        new_folder = f'static/faces/{new_name}_{new_user_id}'
        if (student.name != new_name or student.user_id != new_user_id) and os.path.exists(old_folder):
            try:
                os.rename(old_folder, new_folder)
            except Exception as e:
                flash(f'Error updating face data: {str(e)}', 'danger')
                return redirect(url_for('edit_student', user_id=user_id))

        # Update student details
        student.name = new_name
        student.user_id = new_user_id
        student.username = new_user_id  # Update username to match user_id
        student.password = generate_password_hash(new_user_id)
        student.branch = branch
        db.session.commit()

        # Retrain model
        if os.listdir('static/faces'):
            try:
                train_model()
            except Exception as e:
                flash(f'Error retraining model: {str(e)}', 'warning')

        flash('Student updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_student.html', student=student, branches=BRANCHES)

if __name__ == '__main__':
    app.run(debug=True)