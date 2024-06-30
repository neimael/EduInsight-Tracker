# Core Packages
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import Extract_images_from_video as extract
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_detector = dlib.get_frontal_face_detector()

try:
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
except Exception as e:
    st.error(f"Error loading emotion detection model: {e}")


# Définir les émotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def plot_emotion_pie_chart(emotion_data):
    labels = emotion_data.keys()
    sizes = emotion_data.values()
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Customize legend
    ax.legend(wedges, labels,
              title="Emotions",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    st.pyplot(fig)

def detect_faces(our_image):
    try:
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
        return img, faces 
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        return None, None

def detect_emotions(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Redimensionner l'image à (64, 64)
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Ajouter une dimension pour le canal de couleur
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Ajouter une dimension pour le batch

        preds = emotion_model.predict(roi_gray)[0]
        emotion_label = EMOTIONS[preds.argmax()]
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Afficher l'émotion détectée
        cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    return img

def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    return cartoon

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def detect_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame, faces

def upload_video():
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_name_with_extension = uploaded_file.name  # Save the name of the uploaded file
        video_name, extension = os.path.splitext(video_name_with_extension)  # Get the name without extension
        st.video(uploaded_file)
        
        # Save the uploaded video to the Videos directory
        video_dir = "Videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, video_name_with_extension)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return uploaded_file, video_name, video_name_with_extension, video_path
    else:
        return None, None, None, None

def get_recommendations(emotion_data):
    predominant_emotion = max(emotion_data, key=emotion_data.get)
    recommendations = {
        "happy": {
            "message": "😊 Students are mostly happy. Continue using engaging and interactive activities to maintain this positive environment.",
            "activities": ["Organize fun group projects", "Introduce gamified learning activities", "Celebrate achievements with small rewards"]
        },
        "sad": {
            "message": "😢 Some students seem sad. Consider providing additional support, such as counseling services, and creating a more inclusive and supportive classroom environment.",
            "activities": ["Set up peer support groups", "Implement mindfulness and relaxation exercises", "Have one-on-one check-ins with students"]
        },
        "angry": {
            "message": "😡 There is a level of anger detected. Try addressing any potential conflicts or sources of frustration. Encouraging open communication and conflict resolution activities might help.",
            "activities": ["Hold open forums for students to express concerns", "Introduce conflict resolution workshops", "Implement stress-relief activities like physical exercises or art therapy"]
        },
        "surprised": {
            "message": "😲 Students are frequently surprised. This could be a sign of either positive engagement or confusion. Ensure that the surprises are educational and not confusing.",
            "activities": ["Use surprise quizzes to keep engagement high", "Introduce unexpected but relevant educational videos", "Ensure clarity in instructions to avoid confusion"]
        },
        "scared": {
            "message": "😨 Fear is prevalent. It may be helpful to create a safer and more predictable classroom environment, reduce high-stakes testing, and provide reassurance.",
            "activities": ["Create a more predictable classroom routine", "Provide positive reinforcement and reassurance", "Implement team-building exercises to build trust"]
        },
        "disgust": {
            "message": "🤢 Disgust is noted. Evaluate if there's anything in the content or classroom environment that could be causing this reaction. Address any identified issues promptly.",
            "activities": ["Gather feedback to identify sources of discomfort", "Introduce more engaging and appealing materials", "Promote a positive and respectful classroom culture"]
        },
        "neutral": {
            "message": "😐 Neutral emotions dominate. This might indicate a need for more dynamic and engaging teaching methods to foster a more active learning environment.",
            "activities": ["Incorporate interactive activities like debates and group work", "Use multimedia resources to enhance lessons", "Encourage student-led projects to boost engagement"]
        }
    }

    recommendation = recommendations.get(predominant_emotion, {
        "message": "Keep monitoring the emotions of the students and adjust your teaching strategies as necessary.",
        "activities": []
    })

    return recommendation["message"], recommendation["activities"]

def get_concentration_recommendations(low_concentration_freq, high_concentration_freq):
    if high_concentration_freq > 75:
        message = "😊 Students are highly concentrated. Keep up the good work with engaging and interactive activities."
        activities = ["Continue with current teaching methods", "Introduce advanced topics to challenge students", "Maintain the interactive and dynamic classroom environment"]
    elif low_concentration_freq > 50:
        message = "😐 Students have low concentration levels. Consider using more dynamic and engaging teaching methods to capture their attention."
        activities = ["Incorporate multimedia resources", "Use interactive activities like group discussions and debates", "Implement short breaks to keep students refreshed"]
    else:
        message = "Mixed levels of concentration detected. Adjust your teaching strategies to maintain high concentration levels."
        activities = ["Mix up teaching methods to cater to different students", "Provide positive feedback to highly concentrated students", "Engage less concentrated students with hands-on activities"]

    return message, activities

def main():
    """EduInsight Tracker! 👨‍🏫"""
    st.title("EduInsight Tracker! 👨‍🏫")

    activities = ["Process Images", "Process Videos", "Process CSV files"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Process Images':
        st.subheader("Process Images")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == 'Gray-Scale':
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
                st.image(img, channels='GRAY')
            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)
            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)
            elif enhance_type == 'Blurring':
                new_img = np.array(our_image.convert('RGB'))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
                blur_img = cv2.GaussianBlur(img, (11, 11), int(blur_rate))
                blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
                st.image(blur_img)

            task = ["Faces", "Emotions", "Cannize", "Cartonize"]
            feature_choice = st.sidebar.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == 'Faces':
                    result_img, result_faces = detect_faces(our_image)
                    st.image(result_img)
                    st.success(f"Found {len(result_faces)} faces")
                elif feature_choice == 'Emotions':
                    result_img = detect_emotions(our_image)
                    st.image(result_img)
                elif feature_choice == 'Cartonize':
                    result_img = cartonize_image(our_image)
                    st.image(result_img)
                elif feature_choice == 'Cannize':
                    result_canny = cannize_image(our_image)
                    st.image(result_canny)

    elif choice == 'Process Videos':
        st.subheader("Process Videos")
        # Initialize session state if not already done

        if 'process_running' not in st.session_state:
            st.session_state.process_running = False
            st.session_state.images_extracted = False
        
        uploaded_file, video_name, video_name_with_extension, video_path = upload_video()

        if uploaded_file is not None and video_name_with_extension is not None:
            st.write(f"Selected video name: {video_name_with_extension}")
            output_extracted = f"Extracted_images/{video_name}"  # Output directory
            output_emotions = f"Emotions_detected/{video_name}"
            output_emotions_vd = "Emotions_videos"
            output_data_folder = 'Data_emotions'
            output_concentraions = f"Level_concentration/{video_name}"
            output_concentraions_vd = f"Concentration_videos"
            output_data_folder_concentration = 'Data_concentration'
            
            status_placeholder = st.empty()

            # Check if images have been extracted already
            if not st.session_state.images_extracted:
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we process your video...")
                extract.extract_images(video_path, output_extracted)
                st.session_state.images_extracted = True
                status_placeholder.empty()

            if st.sidebar.button("Detect Emotions !"):
                # Create a placeholder for the status message
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we Detect Emotions of your Students ...")
                # Processing the video
                import Detect_emotions as emotions
                import detect_emotions_vd as emotions_vd
                emotions.process_images(output_extracted, output_emotions)
                # emotions_vd.analyze_video_emotions(video_path, output_emotions_vd, output_data_folder)
                
                status_placeholder.empty()

                # Show images of detected emotions in a 3x3 grid
                st.title("Detected Emotions")
                image_files = os.listdir(output_emotions)

                # Select 9 random images
                random.shuffle(image_files)
                selected_images = image_files[:9]

                # Calculate the number of columns and rows
                num_cols = 3
                num_rows = 3

                # Display images in a 3x3 grid
                columns = st.columns(num_cols)
                for i in range(len(selected_images)):
                    columns[i % num_cols].image(f"Emotions_detected/{video_name}/{selected_images[i]}", caption=selected_images[i], use_column_width=True)

                # Show resulting video
                st.title("Resulting Video")
                video_file_path = os.path.join(output_emotions_vd, f'{video_name}.mp4')
                # emotion_video_path = output_emotions_vd_ld

                # video = os.listdir(output_emotions_vd_ld)

                if os.path.exists(video_file_path):
                    st.video(video_file_path)
                else:
                    st.error(f"Video file not found: {video_file_path}")

                # Read emotion data from text file and display pie chart
                emotion_data = {}
                emotion_file_path = f"{output_data_folder}/{video_name}.txt"
                if os.path.exists(emotion_file_path):
                    with open(emotion_file_path, 'r') as file:
                        for line in file:
                            emotion, percentage = line.strip().split(': ')
                            emotion_data[emotion] = float(percentage.strip('%'))

                    st.title("Emotion Distribution")
                    plot_emotion_pie_chart(emotion_data)

                    # Display recommendations based on emotions
                    st.title("Recommendations for Teachers")
                    recommendation_message, recommendation_activities = get_recommendations(emotion_data)
                    st.write(recommendation_message)
                    if recommendation_activities:
                        st.write("### Suggested Activities:")
                        for activity in recommendation_activities:
                            st.write(f"- {activity}")
                    else:
                        st.error(f"Emotion data file not found: {emotion_file_path}")

            if st.sidebar.button("Detect Level of concentration !"): 
                # Create a placeholder for the status message
                status_placeholder = st.empty()
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we Detect concentration's level of your Students ...")

                # Processing the video
                import Concentration_level as concentraion
                import Concentration_level_vd as concentration_vd

                concentraion.detect_concentration(output_extracted, output_concentraions)
                # concentration_vd.accelerate_video_with_concentration(video_path, output_concentraions_vd, output_data_folder_concentration)

                # Clear the status message
                status_placeholder.empty()

                # Show images of detected concentration levels in a 3x3 grid
                st.title("Detected Concentration's Level")
                image_files = os.listdir(output_concentraions)

                # Select 9 random images
                random.shuffle(image_files)
                selected_images = image_files[:9]

                # Calculate the number of columns and rows
                num_cols = 3
                num_rows = 3

                # Display images in a 3x3 grid
                columns = st.columns(num_cols)
                for i in range(len(selected_images)):
                    columns[i % num_cols].image(f"Level_concentration/{video_name}/{selected_images[i]}", caption=selected_images[i], use_column_width=True)

                # Show resulting video
                st.title("Resulting Video of Concentration's level")
                video_file_path = f"{output_concentraions_vd}/{video_name}.mp4"
                print(video_file_path)
                st.video(video_file_path)

                # Read concentration level counts from the text file
                concentration_file_path = os.path.join(output_data_folder_concentration, f"{video_name}.txt")
                with open(concentration_file_path, 'r') as file:
                    lines = file.readlines()
                    low_concentration_count = int(lines[0].split(': ')[1])
                    high_concentration_count = int(lines[1].split(': ')[1])

                # Calculate the total frames
                total_frames = low_concentration_count + high_concentration_count

                # Calculate the frequency of low and high concentration frames
                low_concentration_freq = low_concentration_count / total_frames * 100  # Convert to percentage
                high_concentration_freq = high_concentration_count / total_frames * 100  # Convert to percentage

                # Create a doughnut chart for the concentration levels
                st.title("Concentration Level Frequencies")
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    [low_concentration_freq, high_concentration_freq],
                    # labels=['Low Concentration', 'High Concentration'],
                    colors=['red', 'green'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.setp(autotexts, size=10, weight="bold")
                plt.legend(wedges, ['Low Concentration', 'High Concentration'], loc="best")
                st.pyplot(fig)

                # Display recommendations based on concentration levels
                st.title("Recommendations for Teachers")
                recommendation_message, recommendation_activities = get_concentration_recommendations(low_concentration_freq, high_concentration_freq)
                st.write(recommendation_message)
                if recommendation_activities:
                    st.write("### Suggested Activities:")
                    for activity in recommendation_activities:
                        st.write(f"- {activity}")
    
    elif choice == "Process CSV files":
        
        st.subheader("Process CSV files")
        
        # Upload CSV file
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df)

            # Select columns
            selected_columns = st.multiselect('Select columns for analysis', df.columns.tolist())
            if selected_columns:
                df_selected = df[selected_columns]
                st.write(df_selected)

                # Choose operation: Clustering or Classification
                operation = st.selectbox("Choose operation", ["Clustering", "Classification"])

                if operation == "Clustering":
                    # Preprocess categorical variables (one-hot encoding)
                    categorical_columns = df_selected.select_dtypes(include=['object']).columns.tolist()
                    if categorical_columns:
                        encoder = OneHotEncoder(sparse=False, drop='first')
                        df_encoded = pd.DataFrame(encoder.fit_transform(df_selected[categorical_columns]))
                        df_encoded.columns = encoder.get_feature_names_out(categorical_columns)
                        df_selected = pd.concat([df_selected.drop(columns=categorical_columns), df_encoded], axis=1)

                    # Preprocess the data
                    scaler = StandardScaler()
                    df_selected_scaled = scaler.fit_transform(df_selected)

                    # Apply K-Means
                    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
                    kmeans = KMeans(n_clusters=n_clusters)
                    df_selected['cluster'] = kmeans.fit_predict(df_selected_scaled)

                    # Visualize clusters with a pie chart
                    cluster_counts = df_selected['cluster'].value_counts()
                    fig, ax = plt.subplots()
                    wedges, texts, autotexts = ax.pie(cluster_counts, autopct='%1.1f%%', startangle=90)

                    # Add legend
                    ax.legend(wedges, cluster_counts.index, title='Cluster', loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

                    # Equal aspect ratio ensures that pie is drawn as a circle.
                    ax.axis('equal')
                    plt.title('Cluster Distribution')
                    st.pyplot(fig)

                    # Display average scores for each cluster
                    cluster_means = df_selected.groupby('cluster').mean()
                    st.write("Average Scores for Each Cluster:")
                    st.write(cluster_means)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    cluster_means.plot(kind='bar', ax=ax)
                    plt.title('Average Scores by Cluster')
                    plt.ylabel('Average Score')
                    plt.xlabel('Cluster')
                    st.pyplot(fig)

                    # Display the DataFrame with cluster labels
                    st.write("Clustered Data:")
                    st.write(df_selected)

                    # Allow teacher to input chosen variables to classify a student
                    st.sidebar.subheader("Classify a new student")
                    input_data = {}
                    for col in selected_columns:
                        if col in categorical_columns:
                            unique_values = df[col].unique()
                            input_data[col] = st.sidebar.selectbox(f"Select {col}", unique_values)
                        else:
                            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)
                    
                    if st.sidebar.button("Classify"):
                        input_df = pd.DataFrame([input_data])
                        
                        # Handle categorical input
                        if categorical_columns:
                            for col in categorical_columns:
                                if col in input_df:
                                    input_df = input_df.join(pd.DataFrame(encoder.transform(input_df[[col]]), columns=encoder.get_feature_names_out([col])))
                                    input_df = input_df.drop(columns=[col])
                            input_df = pd.DataFrame(input_df, columns=df_selected.columns.drop('cluster'))
                        
                        input_df_scaled = scaler.transform(input_df)
                        predicted_cluster = kmeans.predict(input_df_scaled)[0]
                        
                        st.sidebar.write(f"The student belongs to cluster: {predicted_cluster}")
                        st.sidebar.write("Average scores for this cluster:")
                        st.sidebar.write(cluster_means.loc[predicted_cluster])

                elif operation == "Classification":
                    # Allow user to select target column for classification from original dataset
                    target_column = st.selectbox("Select the target column for classification", df.select_dtypes(include=['object']).columns.tolist())
                    if target_column in df.columns:
                        # Include target column in df_selected
                        df_selected[target_column] = df[target_column]

                        # Prepare data for classification
                        X = df_selected.drop(columns=[target_column])
                        y = df_selected[target_column]

                        # Preprocess categorical variables if present (one-hot encoding)
                        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
                        if categorical_columns:
                            encoder = OneHotEncoder(sparse=False, drop='first')
                            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]))
                            X_encoded.columns = encoder.get_feature_names_out(categorical_columns)
                            X = pd.concat([X.drop(columns=categorical_columns), X_encoded], axis=1)

                        # Split data into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                        # Train classifier
                        classifier = RandomForestClassifier(n_estimators=100)
                        classifier.fit(X_train, y_train)

                        # Display pie chart for target column distribution
                        target_counts = df[target_column].value_counts()
                        fig, ax = plt.subplots()
                        wedges, texts, autotexts = ax.pie(target_counts, autopct='%1.1f%%', startangle=90)

                        # Add legend
                        ax.legend(wedges, target_counts.index, title=target_column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

                        # Equal aspect ratio ensures that pie is drawn as a circle.
                        ax.axis('equal')
                        plt.title(f'{target_column} Distribution')
                        st.pyplot(fig)

                        # Allow teacher to input chosen variables to classify a student
                        st.sidebar.subheader("Classify a new student")
                        input_data = {}
                        for col in selected_columns:
                            if col in categorical_columns:
                                unique_values = df[col].unique()
                                input_data[col] = st.sidebar.selectbox(f"Select {col}", unique_values)
                            else:
                                input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)

                        if st.sidebar.button("Classify"):
                            input_df = pd.DataFrame([input_data])

                            # Handle categorical input
                            if categorical_columns:
                                for col in categorical_columns:
                                    if col in input_df:
                                        input_df = input_df.join(pd.DataFrame(encoder.transform(input_df[[col]]), columns=encoder.get_feature_names_out([col])))
                                        input_df = input_df.drop(columns=[col])
                                input_df = pd.DataFrame(input_df, columns=X.columns)

                            # Predict target value for input data
                            predicted_value = classifier.predict(input_df)[0]

                            st.sidebar.write(f"The student's {target_column} is predicted as: {predicted_value}")

                    else:
                        st.sidebar.write(f"Error: '{target_column}' not found in original dataset.")

if __name__ == '__main__':
    main()
