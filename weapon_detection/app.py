import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
# Comment out SQLAlchemy import
# from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import datetime
import json
from models.detector import WeaponDetector

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_development')
# Comment out SQLAlchemy config
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weapon_detection.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
# db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize weapon detector
detector = WeaponDetector()

# Database Models
class User(UserMixin):
    def __init__(self, id, username, email, role='user'):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.password_hash = None
        self.detection_settings = {
            'confidence_threshold': 0.5,
            'alert_sound': 'none',
            'auto_capture': True,
            'show_bounding_boxes': True
        }
        self.notification_settings = {
            'email_notifications': True,
            'high_risk_alerts': True,
            'daily_summary': False
        }
        self.created_at = datetime.datetime.utcnow()
        self.detections = []

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Detection:
    def __init__(self, id, user_id, timestamp=None, image_path=None, confidence=None, weapon_type=None, location=None, notes=None):
        self.id = id
        self.user_id = user_id
        self.timestamp = timestamp or datetime.datetime.utcnow()
        self.image_path = image_path
        self.confidence = confidence
        self.weapon_type = weapon_type
        self.location = location
        self.notes = notes

# Initialize in-memory storage
users = {}
detections = []

# Create admin user
admin = User(1, 'admin', 'admin@example.com', role='admin')
admin.set_password('admin123')
users[1] = admin

# Initialize counters
next_user_id = 2
next_detection_id = 1

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Find user by username
        user = None
        for u in users.values():
            if u.username == username:
                user = u
                break
        
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_user_id
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username exists
        for u in users.values():
            if u.username == username:
                flash('Username already exists.', 'danger')
                return redirect(url_for('register'))
        
        # Check if email exists
        for u in users.values():
            if u.email == email:
                flash('Email already registered.', 'danger')
                return redirect(url_for('register'))
        
        # Create new user
        user = User(next_user_id, username, email)
        user.set_password(password)
        users[next_user_id] = user
        next_user_id += 1
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's detections from in-memory storage
    user_detections = [d for d in detections if d.user_id == current_user.id]
    user_detections.sort(key=lambda x: x.timestamp, reverse=True)
    today_date = datetime.datetime.now().date()
    return render_template('dashboard.html', detections=user_detections, today_date=today_date)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    global next_detection_id
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)
            
            if not allowed_file(file.filename):
                flash('File type not allowed. Please upload an image file (PNG, JPG, JPEG, GIF).', 'danger')
                return redirect(request.url)
            
            # Secure the filename and create filepath
            filename = secure_filename(file.filename)
            
            # Ensure the upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Create a unique filename to avoid overwriting
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            try:
                file.save(filepath)
            except Exception as e:
                flash(f'Error saving file: {str(e)}', 'danger')
                return redirect(request.url)
            
            # Get detection results
            try:
                results = detector.detect_image(filepath)
            except Exception as e:
                flash(f'Error during detection: {str(e)}', 'danger')
                # Clean up the uploaded file if detection fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
            
            # Save detection to database
            try:
                detection = Detection(
                    id=next_detection_id,
                    user_id=current_user.id,
                    image_path=unique_filename,
                    confidence=results[0]['confidence'] if results else 0,
                    weapon_type=results[0]['class'] if results else 'None',
                    location=request.form.get('location', 'Unknown'),
                    notes=request.form.get('notes', '')
                )
                detections.append(detection)
                next_detection_id += 1
            except Exception as e:
                flash(f'Error saving detection record: {str(e)}', 'danger')
                # Clean up the uploaded file if saving fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
            
            return render_template('detection_results.html', 
                                filename=unique_filename, 
                                results=results,
                                now=datetime.datetime.now())
        
        except Exception as e:
            flash(f'An unexpected error occurred: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('detect.html')

@app.route('/detection_results/<filename>')
@login_required
def detection_results(filename):
    # Construct the full file path
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if the file exists
    if not os.path.exists(filepath):
        flash(f'Image file not found: {filename}', 'danger')
        return redirect(url_for('detect'))
    
    try:
        # Process the image
        results = detector.detect_image(filepath)
        return render_template('detection_results.html', 
                              results=results, 
                              filename=filename,
                              now=datetime.datetime.now())
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('detect'))

@app.route('/live_detection')
@login_required
def live_detection():
    return render_template('live_detection.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame with detector
            processed_frame = detector.process_frame(frame)
            
            # Convert to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reports')
@login_required
def reports():
    user_detections = [d for d in detections if d.user_id == current_user.id]
    user_detections.sort(key=lambda x: x.timestamp, reverse=True)
    return render_template('reports.html', detections=user_detections)

@app.route('/delete_detection/<int:detection_id>', methods=['POST'])
@login_required
def delete_detection(detection_id):
    # Find the detection
    detection = next((d for d in detections if d.id == detection_id and d.user_id == current_user.id), None)
    
    if detection:
        # Remove the detection from the list
        detections.remove(detection)
        flash('Detection record deleted successfully.', 'success')
    else:
        flash('Detection record not found or you do not have permission to delete it.', 'danger')
    
    return redirect(url_for('reports'))

@app.route('/delete_all_detections', methods=['POST'])
@login_required
def delete_all_detections():
    # Remove all detections for the current user
    global detections
    detections = [d for d in detections if d.user_id != current_user.id]
    flash('All detection records have been deleted.', 'success')
    return redirect(url_for('reports'))

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        # Verify current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('settings'))
        
        # Check if username is already taken by another user
        for user in users.values():
            if user.id != current_user.id and user.username == username:
                flash('Username already exists.', 'danger')
                return redirect(url_for('settings'))
        
        # Check if email is already taken by another user
        for user in users.values():
            if user.id != current_user.id and user.email == email:
                flash('Email already registered.', 'danger')
                return redirect(url_for('settings'))
        
        # Update user information
        current_user.username = username
        current_user.email = email
        
        # Update password if provided
        if new_password:
            current_user.set_password(new_password)
        
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'An error occurred while updating your profile: {str(e)}', 'danger')
        return redirect(url_for('settings'))

@app.route('/update_detection_settings', methods=['POST'])
@login_required
def update_detection_settings():
    try:
        # Get form data
        confidence_threshold = request.form.get('confidence_threshold')
        alert_sound = request.form.get('alert_sound')
        auto_capture = request.form.get('auto_capture') == 'on'
        show_bounding_boxes = request.form.get('show_bounding_boxes') == 'on'
        
        # Update user's detection settings
        current_user.detection_settings = {
            'confidence_threshold': float(confidence_threshold) / 100,  # Convert to decimal
            'alert_sound': alert_sound,
            'auto_capture': auto_capture,
            'show_bounding_boxes': show_bounding_boxes
        }
        
        flash('Detection settings updated successfully.', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'An error occurred while updating detection settings: {str(e)}', 'danger')
        return redirect(url_for('settings'))

@app.route('/update_notification_settings', methods=['POST'])
@login_required
def update_notification_settings():
    try:
        # Get form data
        email_notifications = request.form.get('email_notifications') == 'on'
        high_risk_alerts = request.form.get('high_risk_alerts') == 'on'
        daily_summary = request.form.get('daily_summary') == 'on'
        
        # Update user's notification settings
        current_user.notification_settings = {
            'email_notifications': email_notifications,
            'high_risk_alerts': high_risk_alerts,
            'daily_summary': daily_summary
        }
        
        flash('Notification settings updated successfully.', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'An error occurred while updating notification settings: {str(e)}', 'danger')
        return redirect(url_for('settings'))

# Admin routes
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin/dashboard.html', users=users.values())

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin/users.html', users=users.values())

@app.route('/admin/add_user', methods=['GET', 'POST'])
@login_required
def add_user():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            role = request.form.get('role', 'user')
            
            # Check if username exists
            for u in users.values():
                if u.username == username:
                    flash('Username already exists.', 'danger')
                    return redirect(url_for('add_user'))
            
            # Check if email exists
            for u in users.values():
                if u.email == email:
                    flash('Email already registered.', 'danger')
                    return redirect(url_for('add_user'))
            
            # Create new user
            user = User(next_user_id, username, email, role=role)
            user.set_password(password)
            users[next_user_id] = user
            next_user_id += 1
            
            flash('User added successfully.', 'success')
            return redirect(url_for('admin_users'))
        
        except Exception as e:
            flash(f'An error occurred while adding the user: {str(e)}', 'danger')
            return redirect(url_for('add_user'))
    
    return render_template('admin/add_user.html')

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user = users.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))
    
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            role = request.form.get('role')
            
            # Check if username is taken by another user
            for u in users.values():
                if u.id != user_id and u.username == username:
                    flash('Username already exists.', 'danger')
                    return redirect(url_for('edit_user', user_id=user_id))
            
            # Check if email is taken by another user
            for u in users.values():
                if u.id != user_id and u.email == email:
                    flash('Email already registered.', 'danger')
                    return redirect(url_for('edit_user', user_id=user_id))
            
            # Update user information
            user.username = username
            user.email = email
            user.role = role
            
            # Update password if provided
            password = request.form.get('password')
            if password:
                user.set_password(password)
            
            flash('User updated successfully.', 'success')
            return redirect(url_for('admin_users'))
        
        except Exception as e:
            flash(f'An error occurred while updating the user: {str(e)}', 'danger')
            return redirect(url_for('edit_user', user_id=user_id))
    
    return render_template('admin/edit_user.html', user=user)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    user = users.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))
    
    try:
        # Remove user's detections
        global detections
        detections = [d for d in detections if d.user_id != user_id]
        
        # Remove user
        del users[user_id]
        
        flash('User deleted successfully.', 'success')
    except Exception as e:
        flash(f'An error occurred while deleting the user: {str(e)}', 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/detections')
@login_required
def admin_detections():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin/detections.html', detections=detections)

@app.route('/admin/settings')
@login_required
def admin_settings():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin/settings.html')

# API endpoints
@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = detector.detect_image(filepath)
        return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True) 