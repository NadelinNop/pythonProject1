import tkinter as tk
from tkinter import messagebox, simpledialog, Label
import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
embedding_model = tf.keras.models.load_model('Embedding_Model.h5')
anti_spoofing_model = tf.keras.models.load_model('Anti_Spoofing_Model.h5')
classifier_model = tf.keras.models.load_model('ClassifierModel.h5')

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Function to capture and process image
def capture_image():
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        messagebox.showerror("Error", "Failed to capture image.")
        return None

# Function to display the captured image
def show_captured_image(frame):
    if frame is not None:
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to check if the person is registered
def verify_identity(image):
    if image is None:
        return

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = preprocess_image(img)

    # Anti-spoofing check
    spoof_check = anti_spoofing_model.predict(img)
    if spoof_check > 0.5:
        messagebox.showerror("Error", "Spoof detected. Please use a real face.")
        status_label.config(text="Status: Spoof detected")
        return

    # Face verification
    embedding = embedding_model.predict(img)
    similarities = []
    for db_img_path in os.listdir('database/'):
        db_img = Image.open(os.path.join('database/', db_img_path))
        db_img = preprocess_image(db_img)
        db_embedding = embedding_model.predict(db_img)
        similarity = cosine_similarity(embedding, db_embedding).item()
        similarities.append((similarity, db_img_path))

    if not similarities:
        messagebox.showinfo("Database Empty", "No registered identities found. Registering new identity.")
        register_identity(image)
        return

    similarities.sort(reverse=True, key=lambda x: x[0])
    for sim, path in similarities:
        print(f"Similarity: {sim} with {path}")

    if similarities[0][0] > 0.98:  # Increase the threshold for similarity
        identity = similarities[0][1].split('.')[0]
        messagebox.showinfo("Identity Verified", f"Identity: {identity}")
        status_label.config(text=f"Status: Identity Verified - {identity}")
    else:
        messagebox.showinfo("Identity Not Found", "Registering new identity.")
        register_identity(image)

# Function to register a new identity
def register_identity(image):
    new_id = simpledialog.askstring("Input", "Enter new identity:")
    if new_id:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img.save(f'database/{new_id}.jpg')
        messagebox.showinfo("Success", "New identity registered successfully.")
        status_label.config(text="Status: New identity registered successfully")

# Function to update the frame in the Label widget
def show_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    lbl.after(10, show_frame)

# Function to capture and verify
def capture_and_verify():
    frame = capture_image()
    show_captured_image(frame)
    verify_identity(frame)

# Function to register a new user
def capture_and_register():
    frame = capture_image()
    if frame is not None:
        register_identity(frame)

# GUI Setup
root = tk.Tk()
root.title("Face Recognition Attendance System")

# Camera setup
cap = cv2.VideoCapture(0)

# UI Elements
frame = tk.Frame(root)
frame.pack()

status_label = Label(root, text="Status: Waiting for input", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

lbl = tk.Label(root)
lbl.pack()

capture_button = tk.Button(frame, text="Capture and Verify Image", command=capture_and_verify)
capture_button.grid(row=0, column=0, padx=10, pady=10)

register_button = tk.Button(frame, text="Register New User", command=capture_and_register)
register_button.grid(row=0, column=1, padx=10, pady=10)

exit_button = tk.Button(frame, text="Exit", command=root.quit)
exit_button.grid(row=0, column=2, padx=10, pady=10)

# Start the video stream
show_frame()

root.mainloop()

# Release camera
cap.release()
cv2.destroyAllWindows()
