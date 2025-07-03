# Face Recognition Based Attendance System

A modern attendance system that uses facial recognition technology to automate the attendance process in educational institutions.

## Features

- **Face Recognition**: Automated attendance marking using facial recognition
- **Multiple User Roles**: Admin, Teacher, and Student interfaces
- **Subject-wise Attendance**: Track attendance for different subjects
- **Real-time Monitoring**: Live attendance tracking
- **Export Options**: Download attendance records in Excel format
- **Secure Authentication**: Role-based access control
- **Responsive Design**: Modern UI that works on all devices

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Interface  │────▶│  Flask Server   │────▶│  Face Detection │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  SQLite DB      │     │  Face Model     │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite
- **Face Recognition**: OpenCV, scikit-learn
- **Data Processing**: Pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mohitradheuniyal/face-recognition-based-attendance-system.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Usage

1. **Admin Access**:
   - Add/Edit/Delete students
   - Manage teachers and subjects
   - View all attendance records

2. **Teacher Access**:
   - Take attendance using face recognition
   - View subject-wise attendance
   - Export attendance records

3. **Student Access**:
   - View personal attendance records
   - Track attendance by subject

## Screenshots

[Add screenshots of your application here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Mohit Radhe Uniyal**
  - GitHub:https://github.com/MohitRadheUniyal/Face-Recognition-Based-Attendance-System-Using-Flask
