# //D:\SEM8\BigData\assi\LyricSense\train_model.py
import os
import sys
import shutil
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Setup environment variables for Java & Hadoop
os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.27.6-hotspot"
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
os.environ["PATH"] = "C:\\Windows\\System32;C:\\Hadoop\\bin;" + os.environ["PATH"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Spark session
try:
    spark = SparkSession.builder \
        .appName("MusicGenreClassification") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.sql.warehouse.dir", "file:///D:/SEM8/BigData/assi/LyricSense/spark-warehouse") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()
    logger.info("Spark session created successfully")
except Exception as e:
    logger.error(f"Failed to create Spark session: {str(e)}")
    sys.exit(1)

# Load the dataset
def load_data():
    try:
        dataset_path = "D:/SEM8/BigData/assi/LyricSense/MendeleyDataset/tcc_ceds_music.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"CSV file not found at: {dataset_path}")

        df = spark.read.csv(f"file:///{dataset_path.replace(os.sep, '/')}", header=True, inferSchema=True)
        logger.info(f"Successfully loaded dataset with {df.count()} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# Build the ML pipeline
def create_pipeline():
    try:
        tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
        idf = IDF(inputCol="raw_features", outputCol="features")
        indexer = StringIndexer(inputCol="genre", outputCol="label")
        nb = NaiveBayes(labelCol="label", featuresCol="features")

        pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, indexer, nb])
        logger.info("Pipeline created successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise

# Main logic
def main():
    try:
        logger.info("Loading data...")
        df = load_data()

        logger.info("Splitting data into train/test sets...")
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Training set size: {train_data.count()}, Test set size: {test_data.count()}")

        logger.info("Creating and training pipeline...")
        pipeline = create_pipeline()
        model = pipeline.fit(train_data)
        logger.info("Model training completed")

        logger.info("Evaluating model...")
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Test Accuracy: {accuracy}")

        logger.info("Saving model...")
        model_path = "D:/SEM8/BigData/assi/LyricSense/trained_model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        model.write().overwrite().save(model_path)
        logger.info("Model saved successfully")

        logger.info("Saving genre mapping...")
        genres = df.select("genre").distinct().collect()
        genre_mapping = {row.genre: idx for idx, row in enumerate(genres)}
        with open("genre_mapping.txt", "w", encoding="utf-8") as f:
            for genre, idx in genre_mapping.items():
                f.write(f"{genre},{idx}\n")
        logger.info("Genre mapping saved")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()
