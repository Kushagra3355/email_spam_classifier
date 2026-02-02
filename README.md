# 📧 Email/SMS Spam Classifier

A machine learning-based spam detection system that classifies emails and SMS messages as spam or legitimate (ham) using Natural Language Processing and various classification algorithms.

Live DEMO: https://spam--classification.streamlit.app/

## 🎯 Project Overview

This project implements a complete spam classification pipeline including:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text preprocessing with NLP techniques
- Multiple machine learning model training and evaluation
- Interactive web application using Streamlit

## 🚀 Features

- **Text Preprocessing Pipeline**
  - Lowercase conversion
  - Tokenization using NLTK
  - Removal of special characters and punctuation
  - Stopwords removal
  - Porter Stemming

- **Machine Learning Models Evaluated**
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - Logistic Regression
  - Support Vector Machine (SVC)
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
  - AdaBoost
  - Bagging Classifier
  - Extra Trees
  - Gradient Boosting
  - XGBoost

- **Interactive Web Interface**
  - Real-time spam detection
  - User-friendly Streamlit interface
  - Instant classification results

## 📋 Requirements

```
numpy
pandas
nltk
scikit-learn
streamlit
xgboost
matplotlib
seaborn
wordcloud
```

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Kushagra3355/email_spam_classifier.git
cd email_spam_classifier
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 Dataset

The project uses the [SMS Spam Collection Dataset](spam.csv) which contains:

- SMS messages labeled as 'spam' or 'ham'
- Preprocessed and cleaned for training

## 🔬 Model Training

The complete model training process is available in [model.ipynb](model.ipynb):

1. **Data Cleaning**
   - Handle missing values
   - Remove unnecessary columns
   - Label encoding

2. **Exploratory Data Analysis**
   - Statistical analysis
   - Visualization of spam vs ham distribution
   - Word cloud generation
   - Correlation analysis

3. **Feature Engineering**
   - Character count
   - Word count
   - Sentence count

4. **Text Preprocessing**
   - Custom `transform_text()` function
   - TF-IDF vectorization

5. **Model Training & Evaluation**
   - Multiple algorithms tested
   - Accuracy and precision metrics
   - Model comparison and selection

## 🎮 Usage

### Running the Web Application

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Using the Classifier

1. Enter your email or SMS message in the text area
2. Click the "Predict" button
3. Get instant classification: **Spam** or **Not Spam**

## 📁 Project Structure

```
email_spam_classifier/
│
├── app.py                 # Streamlit web application
├── model.ipynb           # Jupyter notebook with complete analysis
├── spam.csv              # Dataset
├── model.pkl             # Trained model (pickle file)
├── vectorizer.pkl        # TF-IDF vectorizer (pickle file)
└── README.md             # Project documentation
```

## 🧪 Model Performance

The final model achieves high accuracy in distinguishing between spam and legitimate messages. Detailed performance metrics including accuracy, precision, recall, and F1-score are available in the notebook.

## 🔧 Technical Details

- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing**: NLTK for tokenization and stemming
- **Model Persistence**: Pickle for saving trained model and vectorizer
- **Web Framework**: Streamlit for interactive UI


## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Kushagra3355**

- GitHub: [@Kushagra3355](https://github.com/Kushagra3355)

