# D:\SEM8\BigData\assi\LyricSense\app.py
import os
import sys
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType

# Fix Spark Python worker path
os.environ["PYSPARK_PYTHON"] = r"C:\\Python310\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\\Python310\\python.exe"

# Set environment variables (make sure winutils.exe is in C:\Hadoop\bin)
os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.27.6-hotspot"
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
os.environ["PATH"] = f"C:\\Hadoop\\bin;{os.environ['PATH']}"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Spark session
try:
    spark = SparkSession.builder \
        .appName("LyricGenrePredictor") \
        .master("local[*]") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .config("spark.sql.warehouse.dir", "file:///D:/SEM8/BigData/assi/LyricSense/spark-warehouse") \
        .getOrCreate()
    logger.info("Spark session created.")
    print("Spark version:", spark.version)
except Exception as e:
    logger.error(f"Spark session failed: {str(e)}")
    sys.exit(1)

# Load model
MODEL_PATH = "D:/SEM8/BigData/assi/LyricSense/trained_model"
try:
    model = PipelineModel.load(MODEL_PATH)
    logger.info("Model loaded.")
except Exception as e:
    logger.error(f"Model load failed: {str(e)}")
    sys.exit(1)

# Load genre mapping
genre_map_path = "D:/SEM8/BigData/assi/LyricSense/genre_mapping.txt"
genre_index_to_label = {}
try:
    with open(genre_map_path, "r", encoding="utf-8") as f:
        for line in f:
            genre, idx = line.strip().split(",")
            genre_index_to_label[int(idx)] = genre
    logger.info("Genre mapping loaded.")
except Exception as e:
    logger.error(f"Genre mapping load failed: {str(e)}")
    sys.exit(1)

# Create a UDF to convert probability vector to an array
vector_to_array = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lyrics = data.get("lyrics", "").strip()

        if not lyrics:
            return jsonify({"error": "No lyrics provided"}), 400

        # Create a DataFrame with the lyrics
        df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        
        # Get predictions
        prediction_df = model.transform(df)
        
        # Add a column with probability array
        prediction_df = prediction_df.withColumn("prob_array", vector_to_array(col("probability")))
        
        # Collect results
        row = prediction_df.select("prediction", "prob_array").collect()[0]
        
        # Get predicted genre
        predicted_index = int(row["prediction"])
        predicted_genre = genre_index_to_label.get(predicted_index, "Unknown")
        
        # Create probabilities dictionary
        probability_array = row["prob_array"]
        
        # Check if we have extreme probabilities (all 0s or all 1s)
        max_prob = max(probability_array)
        
        # If the model is super confident (100% on one class), apply softening
        if max_prob > 0.95:
            logger.info("Applying probability softening to avoid 100% predictions")
            # Use softmax to spread the probabilities a bit
            softened_probs = softmax(probability_array)
            probs = {genre_index_to_label[i]: round(float(prob), 4) for i, prob in enumerate(softened_probs)}
        else:
            probs = {genre_index_to_label[i]: round(float(prob), 4) for i, prob in enumerate(probability_array)}
        
        # Return the results
        return jsonify({
            "predicted_genre": predicted_genre,
            "probabilities": probs
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def softmax(x):
    """Apply softmax function to get smoother probability distribution"""
    # Temp parameter controls the "softness" - higher = more confident
    temp = 1.5
    
    # Apply temperature scaling
    x_scaled = np.array(x) / temp
    
    # Subtract max for numerical stability before applying exp
    e_x = np.exp(x_scaled - np.max(x_scaled))
    
    # Return softmax values
    return e_x / e_x.sum()

if __name__ == "__main__":
    app.run(debug=True)