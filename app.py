# D:\SEM8\BigData\assi\LyricSense\app.py
import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Fix Spark Python worker path
os.environ["PYSPARK_PYTHON"] = r"C:\Python310\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Python310\python.exe"

# Set environment variables (make sure winutils.exe is in C:\Hadoop\bin)
os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.27.6-hotspot"
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
os.environ["PATH"] = f"C:\\Hadoop\\bin;{os.environ['PATH']}"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Spark session (disable native IO to prevent winutils errors)
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

        # Create a single-row Spark DataFrame
        df = spark.createDataFrame([(lyrics,)], ["lyrics"])

        # Predict
        prediction_df = model.transform(df)
        predicted_index = int(prediction_df.collect()[0]['prediction'])
        predicted_genre = genre_index_to_label.get(predicted_index, "Unknown")

        # Optionally: simulate uniform probability (or skip if your model doesn't output probability)
        probs = {genre: 1.0 if genre == predicted_genre else 0.0 for genre in genre_index_to_label.values()}

        return jsonify({
            "predicted_genre": predicted_genre,
            "probabilities": probs
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
