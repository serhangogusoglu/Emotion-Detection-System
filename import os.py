import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
import cv2
from playsound import playsound
from deepface import DeepFace
import pickle
import librosa
from nltk import DecisionTreeClassifier
from pydub import AudioSegment
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras as k
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization,Activation,GlobalAveragePooling2D,Dropout,Dense,MaxPooling2D
import random
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# For Natural Language Processing (NLP)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

path_part = r'C:\Users\kerem\Desktop\DuyguAnalizi'

# Load model and vectorizers globally (for fastening text analysis process)
def prepare_model_and_vectorizer():
    model_path = rf"{path_part}\text_analysis\text_analysis_model.pkl"
    vectorizer_path = rf"{path_part}\text_analysis\text_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
    else:
        # Load training set and process
        df = pd.read_csv(rf'{path_part}\text_analysis\tweet_emotions.csv')
        df['cleaned_content'] = df['content'].apply(preprocess_text)
        df['processed_content'] = df['cleaned_content'].apply(tokenize_and_stem)

        sentiment_map = {'empty': 0, 'anger': 1, 'boredom': 2, 'enthusiasm': 3, 'fun': 4, 'happiness': 5, 'hate': 6,
                         'love': 7, 'neutral': 8, 'relief': 9, 'sadness': 10, 'surprise': 11, 'worry': 12}
        df['sentiment'] = df['sentiment'].map(sentiment_map)

        X = df['processed_content']
        y = df['sentiment']

        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

        # Balance datas with SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train a Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_res, y_train_res)

        # Save model and vectorizer
        with open(model_path, "wb") as model_file, open(vectorizer_path, "wb") as vectorizer_file:
            pickle.dump(model, model_file)
            pickle.dump(vectorizer, vectorizer_file)

    return model, vectorizer

# Clean the text
def preprocess_text(text):
    text = re.sub(r"http\S+|@\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text


# Tokenizing and stemming
def tokenize_and_stem(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

class EmotionAnalysisApp(App):
    def build(self):
        self.main_layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        title = Label(
            text='Emotion Analysis Application',
            size_hint_y=None,
            height=50,
            font_size=24
        )
        self.main_layout.add_widget(title)

        self.text_input = TextInput(
            multiline=True,
            size_hint_y=None,
            height=100,
            hint_text='Enter the text you want to analyze...'
        )
        self.main_layout.add_widget(self.text_input)

        button_grid = GridLayout(
            cols=2,
            spacing=10,
            size_hint_y=None,
            height=250,
            padding=10
        )

        buttons = [
            ('Text Analysis', self.text_analysis),
            ('Image Analysis', self.image_analysis),
            ('Audio Analysis', self.audio_analysis),
            ('Camera Analysis', self.camera_analysis),
            ('Video Analysis', self.video_analysis)
        ]

        for text, callback in buttons:
            btn = Button(
                text=text,
                size_hint=(None, None),
                size=(200, 50),
                background_color=(0.2, 0.6, 1, 1)
            )
            btn.bind(on_press=callback)
            button_grid.add_widget(btn)

        self.main_layout.add_widget(button_grid)

        self.result_label = Label(
            text='Results will be shown here...',
            size_hint_y=None,
            height=100
        )
        self.main_layout.add_widget(self.result_label)

        return self.main_layout

    def text_analysis(self, instance):
        text = self.text_input.text

        if not text:
            self.result_label.text = "Enter a text to analyze!"
            return

        # Model and vectorizer are ready globally
        prediction = self.analyze_text(text)
        self.result_label.text = f"Text Analysis Result: {prediction}"

    def image_analysis(self, instance):
        try:
            # Select a random image from predict folder and analyze
            result = self.analyze_images()
            self.result_label.text = f"Image Analysis Result: {result}"
        except Exception as e:
            self.result_label.text = f"Error: {str(e)}"
            print(f"Detailed Error: {e}")

    def audio_analysis(self, instance, folder_path=rf"{path_part}\audio"):
        try:
            audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav'))]
            if not audio_files:
                return "There is no audio file that can be analyzed!"

            audio_path = os.path.join(folder_path, random.choice(audio_files))
            if not os.path.exists(audio_path):
                self.result_label.text = f"'{audio_path}' was not found!"
                return
            result = self.analyze_audio_emotion(audio_path)
            self.result_label.text = f"Audio Analysis Result: {result}"
            playsound(audio_path)

        except Exception as e:
            self.result_label.text = f"Error: {str(e)}"

    def camera_analysis(self, instance):
        try:
            self.analyze_live_emotion()
        except Exception as e:
            self.result_label.text = f"Camera Error: {str(e)}"

    def video_analysis(self, instance):
        try:
            video_path = rf"{path_part}\videos"
            video_number = np.random.randint(0, 8)
            video_path = fr"{video_path}/{video_number}.mp4"
            if not os.path.exists(video_path):
                self.result_label.text = f"'{video_path}' was not found!"
                return
            results = self.analyze_video_emotion(video_path)
            if results:
                self.result_label.text = f"Video Analysis Result:\n{results}"
        except Exception as e:
            self.result_label.text = f"Video Analysis Error: {str(e)}"

    @staticmethod
    def analyze_text(text):
        model, vectorizer = prepare_model_and_vectorizer()

        # Clean the text and vectorize it
        text = preprocess_text(text)
        text = tokenize_and_stem(text)
        text_vector = vectorizer.transform([text])

        # Make a prediction
        prediction = model.predict(text_vector)[0]

        sentiment_map = {
            0: 'Empty', 1: 'Anger', 2: 'Boredom', 3: 'Enthusiasm', 4: 'Fun', 5: 'Happiness', 6: 'Hate', 7: 'Love',
            8: 'Neutral', 9: 'Relief', 10: 'Sadness', 11: 'Surprise', 12: 'Worry'
        }

        return sentiment_map.get(prediction, "Unknown Emotion")

    @staticmethod
    def analyze_images(folder_path=rf"{path_part}\images\gorselanaliz\predict"):
        try:
            # Select a random image from predict directory
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                return "There is no image file that can be analyzed!"

            image_path = os.path.join(folder_path, random.choice(image_files))

            # Load the image and preprocess it
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "Image could not be read!"

            # Resize the image as expected from the model
            img_resized = cv2.resize(img, (48, 48))
            img_array = np.array(img_resized, dtype='float32')
            img_array = img_array.reshape((1, 48, 48, 1))
            img_array = img_array / 255.0

            # Load the pretrained model
            model_path = rf"{path_part}\images\gorselanaliz\emotion_model.h5"
            if not os.path.exists(model_path):
                return "Model file cannot be found!"

            # Load the model
            model = load_model(model_path,
                               custom_objects=None,
                               compile=False)

            # Compile it
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Predict
            predictions = model.predict(img_array, batch_size=1)
            class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
            predicted_class = class_labels[np.argmax(predictions)]

            # Show the image with opening a window
            photo = Image.open(image_path)
            photo.show()  # Open the image with default image opener of the system

            probability = np.max(predictions) * 100
            result = f"Image: {os.path.basename(image_path)}\nEmotion: {predicted_class}\nProbability of the Emotion: %{probability:.2f}"
            return result

        except Exception as e:
            print(f"Detailed Error: {str(e)}")
            return f"An Error Happened During The Training Process: {str(e)}"

    @staticmethod
    def analyze_audio_emotion(audio_path):
        try:
            # Feature extraction from the audio file
            audio, sr = librosa.load(audio_path, duration=3)

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            # Spectral features
            spectral_centers = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

            # Statistical features
            chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)

            # Merge all of features
            features = np.concatenate([
                mfccs_scaled,
                [np.mean(spectral_centers)],
                [np.mean(spectral_rolloff)],
                [np.mean(chroma_stft)]
            ])

            # Load the model and predict
            with open(rf'{path_part}\audio\emotion_model_audio.pkl', 'rb') as f:
                model = pickle.load(f)

            emotion_dict = {
                '01': 'Neutral', '02': 'Calm', '03': 'Happy', '04': 'Sad', '05': 'Angry',
                '06': 'Fearful', '07': 'Disgust', '08': 'Surprised'
            }

            prediction = model.predict([features])[0]
            return emotion_dict[prediction]

        except Exception as e:
            print(f"Audio Analysis Error: {str(e)}")
            return "Neutral"

    @staticmethod
    def analyze_live_emotion():
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                try:
                    face_roi = frame[y:y + h, x:x + w]
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                except:
                    pass

            cv2.imshow('Emotion Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def analyze_video_emotion(video_path):
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not cap.isOpened():
            return "Video could not be opened!"

        frame_count = 0
        emotions_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                try:
                    face_roi = frame[y:y + h, x:x + w]
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    emotions_history.append(emotion)

                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                except Exception as e:
                    continue

            cv2.imshow('Video Emotion Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if emotions_history:
            emotions_count = Counter(emotions_history)
            dominant_emotion = emotions_count.most_common(1)[0][0]
            total = sum(emotions_count.values())

            results = f"The Most Important Emotion: {dominant_emotion}\n\nEmotion Percents:\n"
            for emotion, count in emotions_count.most_common():
                percentage = (count / total) * 100
                results += f"{emotion}: %{percentage:.1f}\n"

            return results
        return "There is no face in the video!"


if _name_ == '_main_':
    EmotionAnalysisApp().run()