import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import emoji
from googleapiclient.discovery import build
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize the sentiment analyzer
sentiment_analyzer = load_sentiment_model()

# Define known positive and negative emojis
positive_emojis = ["❤️", "😊", "👍", "💖", "🥰", "😄", "🤩","🎊","✨","🥳","😁","💝","👌","👏","🙌"]
negative_emojis = ["💔", "😢", "😡", "👎", "😞", "😔"]

# Function to detect emoji sentiment
def detect_emoji_sentiment(text):
    # Check if any emoji in the positive or negative lists are in the text
    for char in text:
        if char in positive_emojis:
            return "Positive"
        elif char in negative_emojis:
            return "Negative"
    return None 

def bert_sentiment(comment):
    
    if not comment.strip():
        return "Neutral"

    # Handle emojis first
    emoji_sentiment = detect_emoji_sentiment(comment)
    if emoji_sentiment:
        return emoji_sentiment

    # Check for phrases that are typically positive
    positive_phrases = ["killed it", "nailed it", "amazing", "awesome", "fantastic", "great"]
    for phrase in positive_phrases:
        if phrase.lower() in comment.lower():
            return "Positive"

    # Truncate input to avoid tokenizer issues
    max_length = sentiment_analyzer.tokenizer.model_max_length
    truncated_comment = comment[:max_length]

    # Use model prediction
    result = sentiment_analyzer(truncated_comment)
    sentiment_label = result[0]['label']
    if sentiment_label == "POSITIVE":
        return "Positive"
    elif sentiment_label == "NEGATIVE":
        return "Negative"
    else:
        return "Neutral"



# YouTube API setup
def get_youtube_comments(channel_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        # Retrieve the latest videos from the channel
        request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()

        # Check if the response contains the 'items' key and it's not empty
        if 'items' not in response or len(response['items']) == 0:
            st.error(f"No data found for the channel with ID {channel_id}. Please verify the channel ID.")
            return []
        
        # Extract the playlist ID of the channel's uploads
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Now, get the videos in the uploads playlist
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=5
        )
        response = request.execute()

        # Check if the response contains the 'items' key and it's not empty
        if 'items' not in response or len(response['items']) == 0:
            st.error(f"No videos found in the uploads playlist for channel {channel_id}.")
            return []

        video_comments = []
        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            
            # Get comments for each video
            comments_request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100  # Get the latest 100 comments for each video
            )
            comments_response = comments_request.execute()

            # Check if the comments response contains 'items' and it's not empty
            if 'items' in comments_response and len(comments_response['items']) > 0:
                # Collect comments from the response
                for comment_thread in comments_response['items']:
                    comment = comment_thread['snippet']['topLevelComment']['snippet']['textDisplay']
                    video_comments.append(comment)
            else:
                st.warning(f"No comments found for video ID {video_id}.")

        return video_comments

    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return []

# Function for single comment sentiment analysis
def single_content():
    st.title("Single Comment")
    st.write("Test your comment here (context-aware sentiment analysis including emojis)")

    user_input = st.text_area("Enter a comment:")
    
    if st.button("Submit"):
        sentiment = bert_sentiment(user_input)
        st.write(f"You submitted: {user_input}")
        st.write(f"Sentiment: {sentiment}")

def browse_content():
    st.title("Browse Content")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the dataset
            data = pd.read_csv(uploaded_file)

            # Check if the required columns are present
            if 'CONTENT' not in data.columns:
                st.error("The dataset must contain a 'CONTENT' column.")
                return
            if 'CLASS' not in data.columns:
                st.error("The dataset must contain a 'CLASS' column.")
                return

            # Display the first 5 rows of the dataset for verification
            st.write("Dataset Overview:")
            st.write(data.head())

            # Apply ensemble sentiment analysis
            data['Sentiment'] = data['CONTENT'].apply(bert_sentiment)

            # Select a specific range of rows to display
            max_rows = len(data)
            row_range = st.slider("Select row range", 0, max_rows - 1, (0, min(1955, max_rows - 1)))
            filtered_data = data.iloc[row_range[0]:row_range[1] + 1]

            st.write("Filtered Dataset with Sentiment Analysis:")
            st.write(filtered_data)

            # Calculate sentiment distribution for filtered data
            sentiment_counts = filtered_data['Sentiment'].value_counts()
            sentiment_percentages = (sentiment_counts / sentiment_counts.sum()) * 100

            # Plot sentiment distribution with percentage annotations
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'grey'])
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')

            # Annotate each bar with percentage text
            for idx, count in enumerate(sentiment_counts):
                percentage = sentiment_percentages[idx]
                ax.text(idx, count + 5, f'{percentage:.1f}%', ha='center', fontsize=12)

            st.pyplot(fig)

            # Define a spam detection function using map
            filtered_data['Spam'] = filtered_data['CLASS'].map({0: "Not Spam", 1: "Spam"})

            # Display dataset with Spam classification
            st.write("Filtered Dataset with Spam Analysis:")
            st.write(filtered_data)

            # Calculate spam distribution for filtered data
            spam_counts = filtered_data['Spam'].value_counts()
            spam_percentages = (spam_counts / spam_counts.sum()) * 100

            # Plot the Spam distribution with percentage annotations
            fig, ax = plt.subplots()
            spam_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_title('Spam Distribution')
            ax.set_xlabel('Spam Classification')
            ax.set_ylabel('Count')

            # Annotate each bar with percentage text
            for idx, count in enumerate(spam_counts):
                percentage = spam_percentages[idx]
                ax.text(idx, count + 5, f'{percentage:.1f}%', ha='center', fontsize=12)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing the file: {e}")


from googleapiclient.discovery import build
import streamlit as st

# Function to get YouTube comments
def get_youtube_comments(channel_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    try:
        # Retrieve the channel's details to get the uploads playlist ID
        request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()

        # Check if the response contains valid channel information
        if 'items' not in response or len(response['items']) == 0:
            st.error(f"No data found for the channel with ID {channel_id}. Please verify the channel ID.")
            return []

        # Extract the playlist ID for the uploads playlist
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Fetch the videos from the uploads playlist
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=5  # Limit to the latest 5 videos
        )
        response = request.execute()

        # Check if the playlist contains videos
        if 'items' not in response or len(response['items']) == 0:
            st.error(f"No videos found in the uploads playlist for channel {channel_id}.")
            return []

        # Collect comments from each video
        video_comments = []
        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']

            # Fetch the comments for the current video
            comments_request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100  # Limit to the latest 100 comments
            )
            comments_response = comments_request.execute()

            # Check if comments exist for the video
            if 'items' in comments_response and len(comments_response['items']) > 0:
                for comment_thread in comments_response['items']:
                    comment = comment_thread['snippet']['topLevelComment']['snippet']['textDisplay']
                    video_comments.append(comment)
            else:
                st.warning(f"No comments found for video ID {video_id}.")

        return video_comments

    except Exception as e:
        st.error(f"An error occurred while fetching data from the YouTube API: {e}")
        return []


def analyze_youtube_comments(channel_id, api_key):
    st.title("YouTube Channel Comment Sentiment Analysis")

    st.write("Fetching comments from the channel... Please wait.")

    try:
        comments = get_youtube_comments(channel_id, api_key)
        if not comments:
            st.write("No comments found.")
            return

        st.write(f"Analyzing sentiment of {len(comments)} comments...")

        
        sentiment_results = [bert_sentiment(comment) for comment in comments]
        sentiment_df = pd.DataFrame({'Comment': comments, 'Sentiment': sentiment_results})

        st.write("Sentiment Analysis Results:")
        st.write(sentiment_df)

        sentiment_counts = sentiment_df['Sentiment'].value_counts()
        sentiment_percentages = (sentiment_counts / sentiment_counts.sum()) * 100

        # Plot sentiment distribution with percentage annotations
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'grey', 'green'])
        ax.set_title('Sentiment Distribution of YouTube Comments')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')

        # Annotate each bar with percentage text
        for idx, count in enumerate(sentiment_counts):
            percentage = sentiment_percentages[idx]
            ax.text(idx, count + 5, f'{percentage:.1f}%', ha='center', fontsize=12)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")



# Main display function
def display_content():
    navbar_choice = st.sidebar.selectbox("Select Menu", ["Single", "Browse", "YouTube"])

    if navbar_choice == "Single":
        single_content()
    elif navbar_choice == "Browse":
        browse_content()
    elif navbar_choice == "YouTube":
        channel_id = st.text_input("Enter the YouTube Channel ID:")
        api_key = st.text_input("Enter your YouTube API Key:")
        
        if st.button("Analyze Comments"):
            if channel_id and api_key:
                analyze_youtube_comments(channel_id, api_key)
            else:
                st.error("Please provide both the Channel ID and API Key.")

# Run the main display function
display_content()
