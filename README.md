# mental-health-predictor
This project builds and deploys a machine learning model using Random Forest and Logistic Regression to assess the mental health risk of tech workers based on workplace environment, support systems, and demographic factors.

## Live Demo

[Click here to try the app](https://mental-health-predictor-hmq7wwuprtvofdthz8tuvk.streamlit.app/)  

## Problem Statement

With rising mental health concerns in the tech industry, there's a growing need for predictive tools to identify at-risk individuals and promote early intervention. This app uses historical survey data to classify whether a person is likely to seek treatment for mental health issues, based on various workplace and personal conditions.

---

## Features

- Interactive Streamlit web app
- Real-time prediction of mental health risk
- Clean visualizations (Age distribution, Gender, Family History, Country)
- Machine Learning models:
  - Random Forest (main model)
  - Logistic Regression (baseline)
- Custom logo and centered layout for better UX
- Fully open-source and reproducible

---

## Files in This Repo

| File                            | Purpose                                                  |
|---------------------------------|----------------------------------------------------------|
| `app.py`                        | Main Streamlit application                               |
| `mental_health.ipynb`           | Jupyter Notebook for data cleaning, EDA, and model training |
| `model.pkl`                     | Trained Random Forest model                              |
| `scaler.pkl`                    | StandardScaler object used for feature scaling           |
| `encoders.pkl`                  | Dictionary of LabelEncoders for categorical variables    |
| `features.json`                 | List of features used in the model                       |
| `Mental_health_in_Tech.png`     | Custom tech-inspired logo for branding                   |
| `requirements.txt`              | Python dependencies for app deployment                   |

---

## How It Works

1. User selects their attributes through dropdowns and sliders
2. Input is encoded and scaled
3. Model predicts likelihood of mental health risk
4. Result is shown with a probability score

---

## Acknowledgements
Dataset from OSMI Mental Health in Tech Survey
Built with Streamlit, Scikit-learn, Pandas, and Matplotlib


