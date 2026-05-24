# Road Accident Risk Prediction

This project aims to predict road accident risk (a continuous value in the range **0 to 1**) using various environmental and road conditions, within the scope of the Kaggle competition **“Playground Series - Season 5, Episode 10.”**

## 🚀 About the Project

A deployed Streamlit app version of this project is available here:
**[Road Accident Risk Prediction App](https://road-accident-risk-bdsr.streamlit.app/)**

## 🛠 Technical Details

- **Dataset:** 517,754 rows (synthetic dataset)
- **Preprocessing:**
  - Categorical variables were transformed using `get_dummies` (One-Hot Encoding).
  - Boolean values were converted to integer type.
  - Features were scaled to the range **[0, 1]** using `MinMaxScaler`.
  
## 📊 Model Performance

Within the scope of this project, several machine learning models were evaluated. **CatBoost** achieved the best overall performance.

| Model | $R^2$ Score | RMSE | MAE |
| --- | --- | --- | --- |
| **CatBoost** | **0.885357** | **0.056263** | **0.043717** |
| XGBRegressor | 0.885049 | 0.056339 | 0.043746 |
| LightGBM | 0.884721 | 0.056419 | 0.043834 |

## 🧠 Key Findings (Feature Importance)

Feature importance analysis shows that the most influential factors are:

- **Speed limit**
- **Lighting at night**
- **Road curvature**
