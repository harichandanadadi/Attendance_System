## Face Recognition Based Attendance System

This is a desktop application built using Python, OpenCV, and Tkinter for an Attendance Management System. It uses a video-based face recognition model to enroll and recognize users and store their attendance.


## Features

User Enrollment: Capture face images and assign a unique ID to each user.

Model Training: Train an LBPH Face Recognizer using the captured dataset.

Face Detection using Viola-Jones Algorithm.

Face Recognition: Identify users through webcam in real-time.

Attendance Logging: Save recognized IDs into a CSV file.

Graphical User Interface: Simple GUI using Tkinter.



## Requirements

Python 3.x

OpenCV (opencv-python and opencv-contrib-python)

Pillow

Pandas

Tkinter (comes with Python)


## Install dependencies:

pip install opencv-python opencv-contrib-python pillow pandas


## Folder Structure

.
├── face.png                      # Image used in GUI
├── recognizer/
│   └── trainingData.yml         # Saved trained model
├── dataSet/                     # Stores captured face images
│   └── user_<id>_<sample>.jpg
├── attendance.csv               # Attendance file
├── main.py                      # Main application script



## Tech Stack & Algorithms Used

Python 3

OpenCV – Image & video processing

Tkinter – Desktop GUI

Pillow – Image rendering in GUI

Pandas – For attendance CSV handling

LBPH (Local Binary Patterns Histogram) – Face recognition algorithm

Viola-Jones Algorithm – Face detection using Haar Cascade Classifier



## How to Run

1. Run the app:

python main.py


2. Click Train:

Enter a numeric ID

System captures 50 face samples using webcam



3. Click Test:

Recognizes faces and saves unique IDs to attendance.csv

Press E to exit recognition



4. Click Exit to close the application.



## Notes

Make sure your webcam is functional

Only numeric IDs are allowed for enrollment

Recognition confidence threshold is set at < 50

Face detection is based on the Viola-Jones framework      
