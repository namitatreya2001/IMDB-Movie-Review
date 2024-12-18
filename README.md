# IMDB Movie Review Sentiment Analysis

This project implements a **sentiment analysis model** to classify IMDB movie reviews as **Positive** or **Negative** using a **Simple RNN** (Recurrent Neural Network) model. The TensorFlow IMDB dataset and text preprocessing techniques are utilized, along with a pre-trained model.

---

## Project Overview

- **Objective**: Predict the sentiment of movie reviews (Positive/Negative).
- **Dataset**: IMDB Movie Review Dataset from TensorFlow.
- **Model**: Simple RNN model trained with Embedding, ReLU activation, and Dense layers.
- **Input**: User-provided movie reviews.
- **Output**: Predicted sentiment and confidence score.

---

## Workflow

The project involves the following steps:

1. **Data Preparation**:
   - Load the IMDB dataset and generate a word index.
   - Preprocess user input using text encoding and padding.

2. **Model**:
   - The pre-trained **Simple RNN** model is loaded, which includes:
     - An **Embedding** layer to handle word embeddings.
     - A **SimpleRNN** layer to capture sequential information.
     - A **Dense** output layer for binary classification.

3. **Prediction**:
   - Convert user-provided input text into encoded and padded sequences.
   - Predict sentiment using the RNN model.
   - Classify the output as **Positive** or **Negative** based on the prediction score.

---

## Technologies Used

| **Library/Tool**      | **Purpose**                                 |
|------------------------|---------------------------------------------|
| `tensorflow==2.15.0`   | Machine learning library for model training |
| `pandas`              | Data manipulation                           |
| `numpy`               | Numerical computations                      |
| `scikit-learn`        | Text preprocessing                          |
| `matplotlib`          | Data visualization                          |
| `tensorboard`         | Model performance tracking                  |
| `streamlit`           | Deployment of the application               |
| `scikeras`            | Keras and scikit-learn compatibility        |

---

##Future Improvements-
1.Replace the Simple RNN with more advanced models like LSTM or GRU for better accuracy.
2.Deploy the model using Docker for containerization.
3.Enhance the Streamlit UI to allow file uploads for batch predictions.

##Live Project Link(Deployed On Streamlit)-https://imdb-movie-review-fzvngdz6wccep49kcgh2wk.streamlit.app/

