# Importing required libraries
import tkinter as tk
from tkinter import Tk, Button, Label, Toplevel
import cv2
import os
import numpy as np
from PIL import ImageTk, Image
import pandas as pd

# Initializing the main Tkinter window
root = Tk()
root.title('Attendance System using Video Based Face Recognition')  # Set the title of the window
root.geometry("1400x800")  # Set the size of the window
file = ''  # Initialize an empty string variable for future use

# Enrollment Process
def train():
    # Nested function to capture face images for training
    def proceed(id1):
        if not id1.isnumeric():  # Check if the entered ID is numeric
            train()  # Recall the train() function to prompt the user again
            return

        # Initialize face detection using Haar cascades
        faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize the camera
        cam = cv2.VideoCapture(0)
        
        # Check if the camera is opened successfully
        if not cam.isOpened():
            print("Cannot open camera")
            exit()

        # Initialize sample counter
        sampleNum = 0
        
        # Start an infinite loop to capture face images
        while (True):
            ret, img = cam.read()  # Capture a frame from the camera
            
            # Check if a frame is received from the camera
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # Convert the color image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale image
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            
            # Loop over each detected face
            for (x, y, w, h) in faces:
                sampleNum += 1  # Increment sample number
                # Save the face image
                cv2.imwrite(f"/home/hari/Downloads/dataSet/user_{id1}_{sampleNum}.jpg", gray[y:y+h, x:x+w])
                # Draw a rectangle around the detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.waitKey(100)  # Wait for 100 milliseconds
            
            # Display the image with detected faces
            cv2.imshow("face", img)
            cv2.waitKey(3)  # Wait for 3 milliseconds
            
            # Break the loop if more than 50 samples are captured
            if sampleNum > 50:
                break
        
        # Release the camera and close all OpenCV windows
        cam.release()
        cv2.destroyAllWindows()

    # Create a top-level window for the training process
    top = Toplevel()
    top.title("Train")  # Set the title of the top-level window
    
    # Load and display an image in the top-level window
    source_image = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(__file__), "face.png")))
    img = Label(top, image=source_image)
    img.pack()
    
    # Create a label asking the user to enter their ID number
    my_label = Label(top, text="Enter ID Number: ")
    my_label.pack()
    
    # Create a text input box for the user to enter their ID
    inputtxt = tk.Text(top, height=1, width=20)
    inputtxt.pack()
    
    # Create a button to proceed with the training
    proceedButton = tk.Button(top, text="Proceed", command=lambda: proceed(inputtxt.get(1.0, "end-1c")), padx=25,
                              pady=10, fg="white", bg="#26a69a")
    proceedButton.pack()
    
    root.mainloop()

# Recognition Process
def test():
    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Set the path where the face images are stored
    path = '/home/hari/Downloads/dataSet'

    # Function to get face images and corresponding IDs from the specified path
    def getImagesWithID(path):
        # List all the image files in the specified path
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        
        # Initialize lists to store faces and IDs
        faces = []
        IDs = []
        
        # Loop over each image file
        for imagePath in imagePaths:
            # Open the image and convert it to grayscale
            faceImg = Image.open(imagePath).convert('L')
            
            # Convert the grayscale image to a numpy array
            faceNp = np.array(faceImg, 'uint8')
            
            # Extract the ID from the image file name
            ID = int(os.path.split(imagePath)[-1].split('_')[1])
            
            # Append the face image and ID to the lists
            faces.append(faceNp)
            IDs.append(ID)
            cv2.waitKey(10)
        
        # Return the lists of IDs and face images
        return IDs, faces

    # Call the getImagesWithID function to get the IDs and face images
    Ids, faces = getImagesWithID(path)
    
    # Train the recognizer with the face images and corresponding IDs
    recognizer.train(faces, np.array(Ids))
    
    # Save the trained model
    recognizer.save('recognizer/trainingData.yml')
    
    # Initialize face detection using Haar cascades
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the camera
    cam = cv2.VideoCapture(0)
    
    # Load the trained model
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainingData.yml")
    
    # Initialize an empty list to store recognized IDs
    ids = []

    # Start an infinite loop to perform face recognition
    while (True):
        ret, img = cam.read()  # Capture a frame from the camera
        
        # Convert the color image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        
        # Loop over each detected face
        for (x, y, w, h) in faces:
            # Perform face recognition
            id, conf = rec.predict(gray[y:y+h, x:x+w])
            
            # If the confidence is less than 50, append the recognized ID to the list
            if conf < 50:
                ids.append(id)
        
        # Display the image with detected faces
        cv2.imshow("face", img)
        
        # Break the loop if 'e' or 'E' is pressed
        if cv2.waitKey(1) == ord('e') or cv2.waitKey(1) == ord('E'):
            break

    # If recognized IDs list is not empty
    if ids:
        # Remove duplicate IDs
        ids = list(set(ids))
        
        # Create a DataFrame to store the recognized IDs
        df = pd.DataFrame()
        df['Ids'] = ids[0:]
        
        # Save the recognized IDs to an Excel sheet
        df.to_csv('attendance.csv', index=False)

    # Release the camera
    cam.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Exit Button code
class QuitButton(Button):
    def __init__(self, parent):
        Button.__init__(self, parent)
        self['text'] = 'Exit'
        self['command'] = parent.destroy
        self['padx'] = 25
        self['pady'] = 10
        self['fg'] = 'white'
        self['bg'] = 'black'
        self.pack()

# UI code (Tkinter)
# Load and display an image in the main window
source_image = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(__file__), "face.png")))
label = Label(root, image=source_image)
label.pack()

# Create buttons for training and testing
train_button = Button(root, text='Train', command=train, padx=25, pady=10, fg="white", bg="Black")
train_button.pack()
test_button = Button(root, text='Test', command=test, padx=25, pady=10, fg="white", bg="#26a69a")
test_button.pack()

# Create an Exit button
QuitButton(root)

# Start the Tkinter main loop
root.mainloop()

