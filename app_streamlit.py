import streamlit as st
import requests
import matplotlib.pyplot as plt
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MusicGenreClassificationWeb") \
    .getOrCreate()

# Load the trained model
try:
    model = PipelineModel.load("trained_model")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load genre mapping
def load_genre_mapping():
    try:
        with open("genre_mapping.txt", "r") as f:
            genre_mapping = {}
            for line in f:
                genre, idx = line.strip().split(",")
                genre_mapping[int(idx)] = genre
        return genre_mapping
    except Exception as e:
        st.error(f"Error loading genre mapping: {str(e)}")
        st.stop()

genre_mapping = load_genre_mapping()

def predict_genre(lyrics):
    try:
        # Create a DataFrame with the input lyrics
        input_data = spark.createDataFrame([(lyrics,)], ["lyrics"])
        
        # Make prediction
        prediction = model.transform(input_data)
        
        # Get probability distribution
        probabilities = prediction.select("probability").collect()[0][0]
        
        # Create response with genre probabilities
        result = {
            genre_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Get the predicted genre (highest probability)
        predicted_genre = max(result.items(), key=lambda x: x[1])[0]
        
        return predicted_genre, result
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    st.title("Music Genre Predictor")
    st.write("Enter the song lyrics below and click 'Predict' to get genre probabilities.")

    # Text area for user to input lyrics
    user_lyrics = st.text_area("Lyrics:", height=200)

    if st.button("Predict"):
        if not user_lyrics.strip():
            st.error("Please enter some lyrics.")
        else:
            with st.spinner("Analyzing lyrics..."):
                predicted_genre, probabilities = predict_genre(user_lyrics)
                
                if predicted_genre and probabilities:
                    # Display the predicted genre
                    st.subheader(f"Predicted Genre: {predicted_genre}")

                    # Create a pie chart of probabilities
                    fig, ax = plt.subplots(figsize=(10, 6))
                    labels = list(probabilities.keys())
                    probs = list(probabilities.values())
                    
                    # Sort by probability
                    sorted_data = sorted(zip(probs, labels), reverse=True)
                    probs, labels = zip(*sorted_data)
                    
                    ax.pie(probs, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    
                    st.pyplot(fig)
                    
                    # Display probabilities in a table
                    st.subheader("Detailed Probabilities")
                    prob_data = {genre: f"{prob*100:.1f}%" for genre, prob in probabilities.items()}
                    st.table(prob_data)
                else:
                    st.error("Failed to make prediction. Please try again.")

if __name__ == "__main__":
    main() 