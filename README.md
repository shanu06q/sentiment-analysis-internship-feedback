# Sentiment Analysis of Internship Feedback

## Project Overview
This project analyzes employee/internship review text and classifies feedback into **Positive, Neutral, and Negative** sentiment classes.

### Objective
Analyze internship/employee feedback to understand positive and negative sentiments and identify areas where satisfaction can be improved.

## Dataset
File: `employee_reviews.csv`

The dataset contains employee review information including `Overall_rating`, `Likes`, `Dislikes`, and workplace satisfaction dimensions.

Because the dataset does not provide an explicit sentiment label, `Overall_rating` is converted into three labels:

- **1–2 → Negative**
- **3 → Neutral**
- **4–5 → Positive**

The rating is used only to create training labels. The model learns from the written `Likes + Dislikes` feedback.

## Machine Learning Approach
1. Data cleaning
2. Combine `Likes` and `Dislikes`
3. Create three sentiment classes from `Overall_rating`
4. 80/20 stratified train-test split
5. TF-IDF feature extraction with unigrams and bigrams
6. Logistic Regression classifier with balanced class weights
7. Evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
8. Interactive prediction for new feedback
9. Save the complete pipeline as `internship_sentiment_model.pkl`

## Results
- Test samples: 4,778
- Accuracy: 0.5580 (55.80%)
- Macro F1-score: 0.4724

The model is intended as an internship demonstration/prototype. The three-class labels are derived from ratings rather than human-annotated sentiment labels, which should be considered when interpreting performance.

## How to Run
Open `Sentiment_Analysis_Internship_Feedback.ipynb` in Google Colab and run the cells from top to bottom. Upload `employee_reviews.csv` when prompted.

The notebook provides an interactive text input where a user can enter new feedback and receive:
- Positive / Neutral / Negative prediction
- Confidence/probability for each class

## Repository Files
- `Sentiment_Analysis_Internship_Feedback.ipynb` — complete Colab notebook
- `employee_reviews.csv` — dataset
- `internship_sentiment_model.pkl` — trained model pipeline
- `confusion_matrix.png` — evaluation visualization
- `sentiment_distribution.png` — class distribution visualization
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `Sentiment_Analysis_Report.docx` — project report
- `Sentiment_Analysis_Report.pdf` — PDF report

## Author
M. Shayan Farooqui
