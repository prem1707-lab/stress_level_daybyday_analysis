# stress_level_daybyday_analysis
it consists of sample projects about data analytics and data science 
 📱 Smartphone Usage & Work Productivity Prediction

## 📌 Project Overview

This project analyzes **smartphone usage behavior** and predicts **Work Productivity Score** using Machine Learning.

The goal is to understand:

* How phone usage affects productivity
* Impact of sleep, stress, and caffeine
* Which habits reduce or improve performance
* Predict future productivity for new users

---

## 📊 Dataset Features

### 🔢 Numeric Features

* Age
* Daily_Phone_Hours
* Social_Media_Hours
* Sleep_Hours
* Stress_Level
* App_Usage_Count
* Caffeine_Intake_Cups
* Weekend_Screen_Time_Hours

### 🔤 Categorical Features

* Gender
* Occupation
* Device_Type

### 🎯 Target Variable

* **Work_Productivity_Score**

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 Project Workflow

### 1️⃣ Data Cleaning

* Checked null values
* Converted categorical → numeric using One-Hot Encoding
* Removed unnecessary columns (User_ID)

### 2️⃣ Feature Engineering

* Screen time patterns
* Usage behavior metrics
* Productivity influencing factors

### 3️⃣ Model Building

* Train/Test split
* Random Forest Regressor
* Prediction of productivity score

### 4️⃣ Visualization

* Correlation heatmap
* Feature importance
* Actual vs Predicted line plot

---

## 📈 Feature Importance (Top Predictors)

![Image](https://www.researchgate.net/publication/360685654/figure/fig2/AS%3A1157135224307712%401652893944181/Feature-importance-bar-charts-for-several-machine-learning-algorithms.png)

![Image](https://www.researchgate.net/publication/384017993/figure/fig2/AS%3A11431281282857456%401728526545583/Feature-importance-plot-of-the-random-forest-model-according-to-variables-weights.png)

![Image](https://lost-stats.github.io/Presentation/Figures/Images/Heatmap-Colored-Correlation-Matrix/heatmap_colored_correlation_matrix_seaborn_python.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2Abrq_vvcnVqsOWoVvsjT0pA.png)

Based on the trained model:

| Feature             | Importance  |
| ------------------- | ----------- |
| Daily Phone Hours   | High impact |
| Weekend Screen Time | High impact |
| Social Media Hours  | High impact |
| App Usage Count     | Medium      |
| Sleep Hours         | Medium      |
| Stress Level        | Medium      |

👉 More screen time & stress → lower productivity
👉 Better sleep → higher productivity

---

## 🧠 Model Used

### Random Forest Regressor

Why?

* Works well on tabular data
* Handles non-linearity
* No scaling required
* High accuracy

Example:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

## ▶️ How to Run

### Step 1 — Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 2 — Run notebook/script

```bash
python main.py
```

or open Jupyter Notebook.

---

## 📁 Project Structure

```
Smartphone-Productivity-Prediction/
│
├── data.csv
├── notebook.ipynb
├── main.py
├── README.md
```

---

## 📊 Sample Visualizations

* Productivity vs Phone Hours
* Feature Importance
* Correlation Heatmap
* Actual vs Predicted Plot

---

## 🔮 Future Improvements

* Try XGBoost / LightGBM
* Hyperparameter tuning
* Deploy with Streamlit Web App
* Real-time prediction dashboard
* Time-series productivity trends

---

## 🎯 Learning Outcomes

From this project, you will learn:

* Data preprocessing
* Handling categorical data
* Feature selection
* Regression models
* Model evaluation
* Data visualization
* End-to-end ML workflow

---

## 👨‍💻 Author

**Prem Prasad**
Data Science & Machine Learning  


