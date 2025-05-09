# Music Genre Classification System

This system uses PySpark's MLlib to classify song lyrics into different music genres. It provides a web interface where users can input lyrics and get genre predictions with probability distributions.

## Prerequisites

- Python 3.7 or higher
- Java 8 or higher (required for PySpark)
- Mendeley Dataset (should be in the MendeleyDataset directory)

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have the Mendeley Dataset in the `MendeleyDataset` directory.

## Usage

1. First, train the model:
```bash
python train_model.py
```
This will:
- Load the Mendeley dataset
- Preprocess the lyrics
- Train a Naive Bayes classifier
- Save the trained model and genre mapping

2. Start the web application:
```bash
python app.py
```

3. Open your web browser and go to `http://localhost:5000`

4. Enter lyrics in the text area and click "Classify Genre" to see the prediction results.

## Features

- Text preprocessing with tokenization and stop word removal
- TF-IDF feature extraction
- Naive Bayes classification
- Interactive web interface with bar chart visualization
- Support for 7 music genres: pop, country, blues, jazz, reggae, rock, and hip hop

## Error Handling

The application includes error handling for:
- Missing model files
- Invalid input
- Server errors

If you encounter any issues, make sure:
1. The model has been trained (run train_model.py)
2. The Mendeley dataset is present
3. All dependencies are installed
