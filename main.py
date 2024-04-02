import cv2
import speech_recognition as sr
import keyboard
import os

def main():
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    # Listen for the "a" key press to start capturing the name
    print("Press 'a' to start capturing the name...")
    keyboard.wait('a')

    # Capture the audio input
    with sr.Microphone() as source:
        print("Listening for name...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)

    try:
        # Recognize the speech using Google Speech Recognition
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        # Extract the name from the speech
        name = extract_name(text)
        print("Detected name:", name)

        # Turn on the front camera and capture image if face is detected
        capture_image(name)

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Error fetching results from Google Speech Recognition service; {0}".format(e))

def extract_name(text):
    # Split the text into words
    words = text.split()

    # Find the index of "is" in the words list
    try:
        is_index = words.index("is")
    except ValueError:
        return "Name not found"  # Handle case where "is" is not found

    # Concatenate the two words after "is" with an underscore between them
    name = "_".join(words[is_index + 1:is_index + 3])
    return name

def capture_image(name):
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create directory for images if it doesn't exist
    os.makedirs("Images", exist_ok=True)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # If a face is detected, capture and save the image
        if len(faces) > 0:
            image_path = os.path.join("Images", f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Face detected! Image saved as {image_path}")
            break

        # Check for key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
