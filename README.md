# Multiple Disease Prediction App

This project is a Streamlit-based web application that predicts the likelihood of multiple diseases using pre-trained machine learning models. The app currently supports predictions for:

- **Chronic Kidney Disease (CKD)**
- **Liver Disease**
- **Parkinson's Disease**

---

## Application Link
- https://multiple-disease-prediction-ribytbbufqxvdsofgbcxr7.streamlit.app/

---

## 🚀 Features
- User-friendly interface for disease prediction.
- Input fields customized for each disease model.
- Real-time predictions powered by pre-trained models hosted on GitHub.

---

## 🛠️ Technologies Used
- **Streamlit**: For the web interface.
- **Scikit-learn**: For building machine learning models.
- **XGBoost**: For advanced model predictions.
- **Python**: Backend programming.
- **GitHub**: Hosting pre-trained models as pickle files.

---

## 📦 Models and Performance
The following machine learning models are used in the app:

1. **CKD Model**: 
   - **Algorithm**: Decision Tree Classifier  
   - **Accuracy**: 99%  
   - **Model file**: [ckd_model.pkl](https://github.com/Barath-dan/Multiple-Disease-Prediction/blob/main/pickle_file/ckd_model.pkl)

2. **Liver Disease Model**: 
   - **Algorithm**: Extra Trees Classifier  
   - **Accuracy**: 82%  
   - **Model file**: [lvr_model.pkl](https://github.com/Barath-dan/Multiple-Disease-Prediction/blob/main/pickle_file/lvr_model.pkl)

3. **Parkinson's Model**: 
   - **Algorithm**: XGBoost Classifier  
   - **Accuracy**: 93%  
   - **Model file**: [pkn_model.pkl](https://github.com/Barath-dan/Multiple-Disease-Prediction/blob/main/pickle_file/pkn_model.pkl)

---

## 🔧 Installation and Setup
Follow the steps below to run the app locally:

### Prerequisites
- Python 3.7 or later
- `pip` package manager
