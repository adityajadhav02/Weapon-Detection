# Weapon Detection System

A comprehensive solution for detecting weapons in images and video streams using computer vision and deep learning.

![Weapon Detection System](weapon_detection/static/images/logo.png)

## Features

- **Image Detection**: Upload images to detect weapons with high accuracy
- **Live Video Detection**: Real-time weapon detection from webcam or video streams
- **User Management**: Secure authentication and role-based access control
- **Detection Reports**: View and analyze detection history
- **Admin Dashboard**: Manage users, view statistics, and configure system settings
- **API Integration**: RESTful API for integration with other systems

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Computer Vision**: OpenCV, YOLOv3
- **Authentication**: Flask-Login
- **Database**: SQLite (in-memory for development)

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- Flask
- Other dependencies listed in requirements.txt

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/weapon-detection.git
   cd weapon-detection
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLOv3 weights and configuration files:
   ```
   # Create a directory for model files
   mkdir -p weapon_detection/models
   
   # Download YOLOv3 weights and config files
   # You'll need to obtain these files separately as they're not included in the repository
   ```

5. Run the application:
   ```
   python weapon_detection/app.py
   ```

6. Access the application at `http://localhost:5000`

## Usage

### User Interface

- **Home**: Landing page with system overview
- **Login/Register**: User authentication
- **Dashboard**: User-specific dashboard with statistics and recent detections
- **Detect**: Upload images for weapon detection
- **Live Detection**: Real-time detection from webcam
- **Reports**: View and manage detection history
- **Settings**: Configure user preferences

### Admin Interface

- **Admin Dashboard**: System-wide statistics and management
- **User Management**: Add, edit, and delete users
- **Detection Management**: View and manage all detections
- **System Settings**: Configure global system settings

## API Endpoints

- `POST /api/detect`: Upload an image for weapon detection
- `GET /api/detections`: Retrieve detection history
- `GET /api/users`: Retrieve user information (admin only)

## Security Considerations

- All user passwords are hashed using Werkzeug's security functions
- Role-based access control for admin functions
- CSRF protection for all forms
- Secure file upload handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv3 model for object detection
- Flask framework for web application
- Bootstrap for frontend design

## Contact

For questions or support, please contact: support@weapondetection.com
