# 🎯 MyAutoJudge: Programming Problem Difficulty Predictor

An intelligent machine learning system that automatically predicts programming problem difficulty based on textual descriptions. Uses both classification (Easy/Medium/Hard) and regression (numerical score 0-10) models with a beautiful web interface.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

## ✨ Key Features

- **🎲 Dual Prediction Models**: Simultaneously predicts both difficulty class (Easy/Medium/Hard) and numerical score (0-10)
- **🤖 Machine Learning Pipeline**: Random Forest-based models with 540+ engineered features
- **🌐 Responsive Web Interface**: Clean, modern UI with real-time AJAX predictions
- **📊 Multiple ML Algorithms Tested**: Random Forest, Logistic Regression, SVM, Gradient Boosting, Linear Regression
- **🔧 Complete Feature Engineering**: TF-IDF, keyword analysis, text statistics, and mathematical features
- **⚡ Production-Ready**: Flask backend with proper error handling and health checks

## 📁 Project Structure

```
MyAutoJudge/
├── app/
│   ├── app.py                        # Flask web application
│   ├── templates/
│   │   └── index.html                # Web interface
│   └── static/
│       └── style.css                 # Responsive styling
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data cleaning & normalization
│   ├── feature_engineering.py        # 540+ feature extraction
│   ├── train_classifier.py           # Classification model training
│   ├── train_regressor.py            # Regression model training
│   └── predict.py                    # Prediction utilities
│
├── models/
│   ├── classifier_random_forest.pkl  # Trained classification model
│   ├── regressor_random_forest.pkl   # Trained regression model
│   ├── feature_extractor_classifier.pkl
│   ├── feature_extractor_regressor.pkl
│   ├── classifier_comparison.png     # Model comparison visualization
│   └── regressor_comparison.png      # Regression model comparison
│
├── data/
│   ├── problems_data.jsonl           # Raw dataset (JSONL format)
│   └── processed_data.csv            # Preprocessed data
│
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishnubishnoi17/MyAutoJudge.git
   cd MyAutoJudge
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python --version  # Should be 3.8+
   pip list          # Verify all packages
   ```

### Run the Application

**Option 1: Quick Start (Pre-trained Models)**

```bash
python app/app.py
```
Then open your browser to: **http://localhost:5000**

**Option 2: Train Models from Scratch**

```bash
# Step 1: Preprocess data
python src/data_preprocessing.py
# Output: data/processed_data.csv

# Step 2: Train classification model
python src/train_classifier.py
# Output: models/classifier_random_forest.pkl

# Step 3: Train regression model
python src/train_regressor.py
# Output: models/regressor_random_forest.pkl

# Step 4: Test predictions
python src/predict.py

# Step 5: Launch web app
python app/app.py
```

## 📊 Classification Model Performance

### Selected Model: **Random Forest**
**Accuracy: 51.15%**

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Easy** | 0.60 | 0.24 | 0.34 | 153 |
| **Hard** | 0.53 | 0.86 | 0.66 | 389 |
| **Medium** | 0.37 | 0.17 | 0.23 | 281 |
| **Accuracy** | - | - | **0.51** | **823** |
| **Macro Avg** | 0.50 | 0.43 | 0.41 | 823 |
| **Weighted Avg** | 0.49 | 0.51 | 0.46 | 823 |

### Confusion Matrix (Random Forest)

```
Predicted:  Easy  Hard  Medium
Actual Easy:  37    77     39
Actual Hard:  10   336     43
Actual Med:   15   218     48
```

**Interpretation:**
- Hard problems are predicted with 86% recall (336/389 correctly identified)
- Easy problems have lower recall (24%), often misclassified as Hard
- Medium class is challenging to predict accurately (17% recall)

### Model Comparison: Classification

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|----------|-------------------|----------------|------------------|
| **Random Forest** | **0.5115** | **0.50** | **0.43** | **0.41** |
| Logistic Regression | 0.4885 | 0.43 | 0.41 | 0.37 |
| SVM | 0.4848 | 0.48 | 0.35 | 0.25 |

**Detailed Per-Class Performance:**

**Logistic Regression (Accuracy: 48.85%)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Easy | 0.40 | 0.26 | 0.32 | 153 |
| Hard | 0.52 | 0.86 | 0.65 | 389 |
| Medium | 0.35 | 0.10 | 0.16 | 281 |

Confusion Matrix (LR):
```
Predicted:  Easy  Hard  Medium
Actual Easy:  40    88     25
Actual Hard:  29   334     26
Actual Med:   30   223     28
```

**Key Findings:**
- Random Forest **outperforms Logistic Regression by 2.3%** (51.15% vs 48.85%)
- Random Forest **outperforms SVM by 2.67%** (51.15% vs 48.48%)
- Logistic Regression: Better at Medium recall (10%) but worse than RF overall
- SVM: Worst performance, especially with Easy class (2% recall)
- Hard class consistently easiest to predict across all models (86% recall)
- **Random Forest is the clear winner** for this classification task

---

## 📈 Regression Model Performance

### Selected Model: **Random Forest**
**Best MAE: 1.6837**

### Detailed Regression Metrics

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | **1.6837** | **2.0175** | **0.1520** |
| Gradient Boosting | 1.6923 | 2.0316 | 0.1402 |
| Linear Regression | 1.7232 | 2.0992 | 0.0820 |

### Model Performance Analysis

**Random Forest (Selected):**
- Mean Absolute Error: 1.6837 (predictions off by ~1.68 points on 0-10 scale)
- RMSE: 2.0175 (accounts for larger errors)
- R² Score: 0.1520 (model explains 15.2% of variance)

**Gradient Boosting (Close Second):**
- MAE: 1.6923 (slightly worse than RF)
- RMSE: 2.0316
- R² Score: 0.1402
- Status: Competitive alternative, ~0.67% higher MAE

**Linear Regression (Baseline):**
- MAE: 1.7232 (performs worst)
- R² Score: 0.0820 (explains only 8.2% of variance)
- Status: Adequate baseline for comparison

### Dataset Statistics

- **Training Set Size**: 3,289 problems
- **Test Set Size**: 823 problems
- **Score Range**: 1.10 - 9.70 (on 0-10 scale)
- **Features**: 540 engineered features

---

## 🧠 Feature Engineering (540 Features)

### 1. **Text Statistics** (7 features)
- Character count
- Word count
- Average word length
- Sentence count
- Problem description length
- Input description length
- Output description length

### 2. **Mathematical Features** (4 features)
- Math operators count (+, -, ×, ÷, =, <, >)
- Parentheses/brackets count
- Number count
- Formula presence indicator

### 3. **Keyword Detection** (29 features)
Algorithm-specific keywords:
- Graph algorithms: `graph`, `tree`, `dfs`, `bfs`, `dijkstra`
- Dynamic Programming: `dynamic`, `dp`, `recursion`, `memoization`
- Advanced Techniques: `backtrack`, `greedy`, `two-pointer`
- Data Structures: `array`, `string`, `linked-list`, `stack`, `queue`, `hash-table`, `matrix`
- String Operations: `substring`, `subsequence`, `permutation`
- Optimization: `maximum`, `minimum`, `optimize`, `shortest-path`

### 4. **TF-IDF Vectorization** (500 features)
- Bi-gram text features
- Semantic problem understanding
- Captures unique problem terminology

---

## 💻 Technology Stack

### Backend & ML
- **Python 3.8+** - Core language
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning models
- **pandas 2.1.4** - Data manipulation
- **numpy 1.26.2** - Numerical computing
- **joblib 1.3.2** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern responsive design
- **Vanilla JavaScript** - AJAX for async predictions
- **Responsive Design** - Mobile & desktop support

---



## 📖 Dataset Format

The dataset (`data/problems_data.jsonl`) contains competitive programming problems in JSONL format:

```json
{
  "title": "Two Sum",
  "description": "Given an array of integers nums and an integer target...",
  "input_description": "Array of integers and target integer",
  "output_description": "Indices of two numbers that add up to target",
  "problem_class": "hard",
  "problem_score": 5.45
}
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Problem name |
| `description` | string | Full problem statement |
| `input_description` | string | Input format specification |
| `output_description` | string | Expected output format |
| `problem_class` | string | Difficulty: "easy", "medium", or "hard" |
| `problem_score` | float | Numerical difficulty (0-10 scale) |

---

## 🧪 Testing

### Test the Prediction Module
```bash
python src/predict.py
```

### Test with cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Two Sum",
    "description": "Given an array of integers, return indices of two numbers that add up to target",
    "input_description": "Array and target integer",
    "output_description": "Two indices"
  }'
```

---

## 🔧 Troubleshooting

### Models Not Loading
```bash
# Verify model files exist
ls -la models/*.pkl
# Should display 4 files:
# - classifier_random_forest.pkl
# - regressor_random_forest.pkl
# - feature_extractor_classifier.pkl
# - feature_extractor_regressor.pkl
```

### Port 5000 Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or run Flask on different port
python app/app.py --port 8000
```

### Virtual Environment Issues
```bash
# Recreate virtual environment from scratch
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Import Errors
```bash
# Ensure you're in project root directory
cd MyAutoJudge

# Verify src/ folder is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Then run the app
python app/app.py
```

---

## 🎨 Web Interface Features

- ✅ Real-time predictions via AJAX
- ✅ Loading animations during processing
- ✅ Probability visualization with progress bars
- ✅ Error handling with user-friendly messages
- ✅ Fully responsive mobile design
- ✅ Form validation (description required)
- ✅ Clear button to reset form
- ✅ Smooth scrolling to results

---

## 📈 Future Enhancements

- [ ] Advanced NLP models (BERT embeddings, GPT-based features)
- [ ] Address class imbalance using SMOTE or class weights
- [ ] Hyperparameter optimization with GridSearchCV/RandomizedSearchCV
- [ ] User authentication and prediction history
- [ ] Batch prediction (CSV upload)
- [ ] Cloud deployment (Heroku, AWS, Google Cloud)
- [ ] Model explainability (SHAP, LIME)
- [ ] Automated model retraining API
- [ ] A/B testing framework for model versions
- [ ] Database integration (PostgreSQL)
- [ ] Docker containerization
- [ ] Improve Medium class prediction accuracy

---

## 🐛 Known Issues & Limitations

### Classification Challenges
- **Overall Accuracy: 51.15%** - Moderate due to class imbalance
- **Easy Class Issue**: Low recall (24%) - often misclassified as Hard
- **Medium Class Issue**: Very low recall (17%) - hardest to predict
- **Hard Class**: Performs well (86% recall) due to 47% of dataset being Hard problems

### Class Distribution Problem
- Easy: 18.6% of dataset (153 samples)
- Hard: 47.2% of dataset (389 samples)
- Medium: 34.1% of dataset (281 samples)
- **Imbalance Ratio: 2.5:1 (Hard:Easy)**

### Regression Limitations
- **R² Score: 0.1520** - Model explains only 15.2% of variance
- **MAE: 1.6837** - Average prediction error of ±1.68 points on 0-10 scale
- Suggests difficulty scoring is more nuanced than text features alone can capture

### Other Limitations
- Limited to English language problem descriptions
- No support for non-text problem features (e.g., time/space constraints, examples)
- Performance depends heavily on description quality
- May not capture implicit difficulty indicators

---

## 💡 Recommendations for Improvement

1. **Address Class Imbalance**:
   - Use SMOTE (Synthetic Minority Over-sampling Technique)
   - Apply class weights in Random Forest
   - Collect more Easy and Medium problem examples

2. **Enhance Feature Engineering**:
   - Add constraint-based features (time limits, memory)
   - Include example complexity features
   - Use pre-trained embeddings (Word2Vec, GloVe, FastText)

3. **Model Improvements**:
   - Experiment with ensemble methods (Stacking, Voting)
   - Try advanced models (XGBoost, LightGBM)
   - Implement cross-validation for robust evaluation

4. **Data Quality**:
   - Clean and validate problem descriptions
   - Remove duplicate or similar problems
   - Balance dataset across classes

---

## 👨‍💻 Development

### Project Standards
- Python code follows PEP 8 style guidelines
- All models saved with `.pkl` extension using joblib
- Feature extractors trained and saved with models
- Data preprocessing is idempotent

### Key Dependencies
See `requirements.txt` for complete list with versions.

---

## 👥 Contributors

- **[vishnubishnoi17](https://github.com/vishnubishnoi17)** - Project Creator & Developer

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---



---

## 📞 Contact & Support

- **GitHub Profile**: [@vishnubishnoi17](https://github.com/vishnubishnoi17)
- **Repository**: [MyAutoJudge](https://github.com/vishnubishnoi17/MyAutoJudge)
- **Issues**: [GitHub Issues](https://github.com/vishnubishnoi17/MyAutoJudge/issues)

---

## 📚 Additional Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Handling Imbalanced Data](https://imbalanced-learn.org/)

---

**⭐ If you find this project helpful, please star it!**
   AUTHOR :- Vishnu Bishnoi
   Enrollment No. - 24114106
   Contact :- 7297052429

**💡 Have suggestions or found a bug? Open an [issue](https://github.com/vishnubishnoi17/MyAutoJudge/issues)!**

Built with ❤️ for automatic programming problem difficulty assessment
