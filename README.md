# Glushko-Taras-M5-Assignment 
### Final Grade Prediction using SVR

This project uses **Support Vector Regression (SVR)** to predict **final grades (G3)** of students based on various academic and demographic features from the **UCI Student Performance dataset**.

### 📸 Confusion Matrix Output
Here’s the confusion matrix visualization from the model:

![Confusion Matrix](Matrix%20output.png)


## 📌 Features
- Fetches **student performance data** from the UCI ML Repository.
- Cleans and preprocesses data (handling missing values, encoding categorical variables).
- Trains an **SVR model** to predict students' final grades.
- Evaluates model performance using:
  - **Mean Squared Error (MSE)**
  - **Explained Variance Score**
  - **Confusion Matrix & Classification Report**
- Visualizes classification results with **Matplotlib**.

---

## 🚀 Installation
Make sure you have Python installed (recommended version **3.9.6**). Then, install the required dependencies:

```bash
python3 -m pip install numpy pandas matplotlib scikit-learn ucimlrepo
