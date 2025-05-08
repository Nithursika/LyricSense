# D:/SEM8/BigData/assi/LyricSense/train_model.py
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

os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.27.6-hotspot"
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
os.environ["PATH"] = "C:\\Windows\\System32;C:\\Hadoop\\bin;" + os.environ["PATH"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    spark = SparkSession.builder \
        .appName("MusicGenreClassification") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.sql.warehouse.dir", "file:///D:/SEM8/BigData/assi/LyricSense/spark-warehouse") \
        .getOrCreate()
    logger.info("Spark session created successfully")
except Exception as e:
    logger.error(f"Failed to create Spark session: {str(e)}")
    sys.exit(1)

def load_data():
    try:
        dataset_path = "D:/SEM8/BigData/assi/LyricSense/Merged_dataset.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"CSV not found at: {dataset_path}")

        df = spark.read.csv(f"file:///{dataset_path.replace(os.sep, '/')}", header=True, inferSchema=True)
        df = df.select("artist_name", "track_name", "release_date", "genre", "lyrics") \
               .dropna(subset=["lyrics", "genre"]) \
               .filter(col("lyrics") != "")

        logger.info(f"Loaded dataset with {df.count()} valid rows")
        return df
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        raise

def create_pipeline():
    try:
        tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
        idf = IDF(inputCol="raw_features", outputCol="features")
        indexer = StringIndexer(inputCol="genre", outputCol="label")
        nb = NaiveBayes(labelCol="label", featuresCol="features")

        pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, indexer, nb])
        return pipeline
    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}")
        raise

def main():
    try:
        df = load_data()

        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Train size: {train_data.count()}, Test size: {test_data.count()}")

        pipeline = create_pipeline()
        model = pipeline.fit(train_data)

        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        model_path = "D:/SEM8/BigData/assi/LyricSense/trained_model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        model.write().overwrite().save(model_path)
        logger.info("Model saved.")

        # Save genre mapping
        label_mapping = model.stages[-2].labels  # StringIndexer labels
        with open("genre_mapping.txt", "w", encoding="utf-8") as f:
            for idx, genre in enumerate(label_mapping):
                f.write(f"{genre},{idx}\n")
        logger.info("Genre mapping saved.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("Spark session closed")

if __name__ == "__main__":
    main()
