# Emotion Recognition

This project focuses on recognizing emotions from text data using machine learning. The model classifies sentences into categories like `anger`, `hate`, `neutral`, and more, depending on the dataset's labels.

---

## Features

- **Text Preprocessing**: Cleans and tokenizes the text data.
- **Deep Learning Model**: An LSTM-based neural network for emotion classification.
- **Customizable**: Easy to modify for additional labels or datasets.
- **Real-Time Prediction**: Test the model with custom sentences.
- **High Accuracy**: Achieved an accuracy of 99.7% on the test dataset.
---

## Prerequisites

### Software Requirements

- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `tensorflow`
  - `sklearn`

Install the required libraries using the following command:

```bash
pip install pandas numpy tensorflow scikit-learn
```

---

## Dataset

The dataset must be a CSV file with the following structure:

| Unnamed: 0 | text                                     | Emotion |
|------------|-----------------------------------------|---------|
| 0          | I seriously hate one subject to death   | hate    |
| 1          | I am so full of life I feel appalled    | neutral |
| 2          | I’ve been really angry with someone      | anger   |

### Columns:
- **`text`**: The sentence or phrase for classification.
- **`Emotion`**: The corresponding emotion label.

---

## How to Run

### 1. Data Preprocessing

Load and preprocess the dataset:
```python
import pandas as pd

# Load dataset
data = pd.read_csv("emotions.csv")
data.dropna(inplace=True)
```

### 2. Train the Model

Train the model using the LSTM architecture:
```bash
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=50),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(labels.shape[1], activation='softmax')
])
```

### 3. Test the Model

Run the following script to test a sentence:
```python
from tensorflow.keras.models import load_model

model = load_model('emotion_recognition_model.h5')
sentence = "I am so excited about the upcoming vacation!"
# Function to preprocess and predict is provided in the main script.
```

---


## Example Usage

1. Input: `I am feeling so happy today!`
2. Predicted Emotion: `joy`

---

## Future Enhancements

- Use a larger and more diverse dataset.
- Integrate pre-trained models like BERT for better accuracy.
- Deploy the model as a REST API for broader usability.

---

## Contribution

Feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

