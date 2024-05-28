import streamlit as st
import Extract_images_from_video as extract
import os
import matplotlib.pyplot as plt

# Configuration de la page
# st.set_page_config(
#     page_title="EduInsight Tracker",
#     page_icon="📊"  # Utilisez un emoji Unicode pour l'icône de l'application
# )

def upload_video():
    st.sidebar.title("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_name_with_extension = uploaded_file.name  # Save the name of the uploaded file
        video_name = os.path.splitext(video_name_with_extension)[0]  # Get the name without extension        
        st.video(uploaded_file)
        return uploaded_file, video_name, video_name_with_extension
    else:
        return None, None, None

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

def main():
    # Initialize session state if not already done
    if 'process_running' not in st.session_state:
        st.session_state.process_running = False
        st.session_state.images_extracted = False

    # Call the function to handle video upload
    uploaded_file, video_name, video_name_with_extension = upload_video()
    
    if uploaded_file is not None and video_name_with_extension is not None:
        st.write(f"Selected video name: {video_name_with_extension}")
        video_path = f"Videos/{video_name_with_extension}"  # Assuming Videos directory exists
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

            # Clear the status message
            status_placeholder.empty()

            # Show images of detected emotions in a 3x3 grid
            st.title("Detected Emotions")
            image_files = os.listdir(output_emotions)
            num_images = len(image_files)
            num_cols = 3
            num_rows = (num_images + num_cols - 1) // num_cols

            columns = st.columns(num_cols)
            for i in range(num_images):
                columns[i % num_cols].image(f"Emotions_detected/{video_name}/{image_files[i]}", caption=image_files[i], use_column_width=True)

            # Show resulting video
            st.title("Resulting Video")
            video_file_path = f"{output_emotions_vd}/{video_name}.mp4"
            print(video_file_path)
            st.video(video_file_path)

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

            # Show images of detected concentration in a 3x3 grid
            st.title("Detected Concentration's level")
            image_files = os.listdir(output_concentraions)
            num_images = len(image_files)
            num_cols = 3
            num_rows = (num_images + num_cols - 1) // num_cols

            columns = st.columns(num_cols)
            for i in range(num_images):
                columns[i % num_cols].image(f"Level_concentration/{video_name}/{image_files[i]}", caption=image_files[i], use_column_width=True)

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

            
            
if __name__ == "__main__":
    main()
