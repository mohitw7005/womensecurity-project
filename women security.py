# Define paths to your dataset folders
female_dir = 'Female'
male_dir = 'male'

# Preprocess and load the dataset
def load_data(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)
                labels.append(label)
    return images, labels
# Create a simple CNN model
def create_gender_recognition_model(input_shape):
    model = tf.keras.Sequential()
    
    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Flatten the output for dense layers
    model.add(tf.keras.layers.Flatten())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # Add dropout to reduce overfitting
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary gender classification
    
    return model

# Create the model
model = create_gender_recognition_model((100, 100, 3))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10)

# Evaluate the model on the validation set
test_loss, test_acc = model.evaluate(datagen.flow(X_val, y_val, batch_size=32))
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('gender_model2.h5')
gender_model = load_model("gender_model2.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Convert the frame to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    face = frame[y:y+h, x:x+w]

    # Preprocess the face image for gender classification
    face = cv2.resize(face, (100, 100))  # Resize to (100, 100)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0

    # Predict the gender
    gender_prediction = gender_model.predict(face)
    gender = "Female" if gender_prediction[0][0] < 0.62 else "Male
# Determine the text color and generate a warning beep if the gender is "Male"
if gender == "Male":
    text_color = (0, 0, 255)  # Red color for "Male"
    winsound.Beep(1000, 500)  # Generate a beep sound (1000 Hz for 500 ms)
else:
    text_color = (0, 255, 0)  # Green color for "Female"

# Draw a rectangle around the detected face and display the gender with the determined text color
cv2.rectangle(frame, (x, y), (x+w, y+h), text_color, 2)
cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

# Display the frame with detected faces and gender classification
cv2.imshow("Gender Detection", frame)

