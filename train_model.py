import os
import sys
import shutil
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer,HashingTF
)
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

# --- Environment Setup for Windows & Spark ---
os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.27.6-hotspot"
os.environ["HADOOP_HOME"] = "C:\\Hadoop"
os.environ["PATH"] = "C:\\Windows\\System32;C:\\Hadoop\\bin;" + os.environ["PATH"]
os.environ["HADOOP_OPTS"] = "-Djava.library.path="  # Disable native IO to prevent access0 error

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Spark Session Initialization ---
try:
    spark = SparkSession.builder \
        .appName("MusicGenreClassification") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
        .getOrCreate()
    logger.info("Spark session created successfully")
except Exception as e:
    logger.error(f"Failed to create Spark session: {str(e)}")
    sys.exit(1)

# --- Load Data Function ---
def load_data():
    try:
        dataset_path = "D:/SEM8/BigData/assi/LyricSense/MendeleyDataset/tcc_ceds_music.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = spark.read.csv(f"file:///{dataset_path.replace(os.sep, '/')}", header=True, inferSchema=True)
        logger.info(f"Successfully loaded dataset with {df.count()} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# --- Create ML Pipeline ---
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

# --- Main Execution ---
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Spark session created successfully")
    spark = SparkSession.builder.appName("LyricsGenreClassification").getOrCreate()

    try:
        logger.info("Loading data...")
        df = spark.read.csv("MendeleyDataset/tcc_ceds_music.csv", header=True, inferSchema=True)
        logger.info(f"Successfully loaded dataset with {df.count()} rows")

        logger.info("Splitting data...")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Training set size: {train_df.count()}, Test set size: {test_df.count()}")

        logger.info("Training model...")

        tokenizer = Tokenizer(inputCol="lyrics", outputCol="tokens")
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=262144)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="genreIndex", maxIter=10)

        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
        logger.info("Pipeline created successfully")

        # Index label
        from pyspark.ml.feature import StringIndexer
        indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
        indexed_train_df = indexer.fit(train_df).transform(train_df)
        indexed_test_df = indexer.fit(test_df).transform(test_df)

        model = pipeline.fit(indexed_train_df)
        logger.info("Model training completed")

        logger.info("Evaluating model...")
        predictions = model.transform(indexed_test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol="genreIndex", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        logger.info("Saving model...")
        try:
            model.save("output/lyrics_genre_model")
            print("✅ Model saved successfully to 'output/lyrics_genre_model'")
        except Exception as e:
            print("❌ ERROR: Model saving failed.")
            print("Reason:", str(e))
            print("\n📋 You can copy the model information below for manual use:\n")
            print("🔍 Model Summary:")
            for idx, stage in enumerate(model.stages):
                print(f"--- Stage {idx+1}: {stage.__class__.__name__} ---")
                print(stage)

    except Exception as e:
        logger.error("An error occurred: %s", str(e))

    finally:
        logger.info("Spark session stopped.")
        spark.stop()

if __name__ == "__main__":
    main()