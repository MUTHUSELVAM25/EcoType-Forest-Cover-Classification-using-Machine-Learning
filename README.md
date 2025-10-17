# 🌿 EcoType: Forest Cover Classification using Machine Learning  

### 🧠 Domain: Environmental Data & Geospatial Predictive Modeling  
**Author:** Muthu Selvam  

---

## 📋 Skills & Tools Applied

- 🧹 Data Cleaning & Preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- ⚙️ Feature Engineering & Selection  
- 🧠 Machine Learning Model Building  
- 🎯 Model Evaluation & Hyperparameter Tuning  
- 🌐 Streamlit Web App Development  
- 💾 Model Deployment (.pkl file)  

---

## 🎯 Project Objective

To develop a **machine learning classification model** that predicts the **forest cover type** in a geographical area based on cartographic variables like **elevation, slope, soil type, and wilderness area**.  

The system assists in **environmental monitoring, forest resource management, and land-use planning** by providing accurate and automated forest cover predictions.

---

## 🌍 Real-World Use Cases

- **Forest Resource Management:** Classify forest areas for conservation or logging.  
- **Wildfire Risk Assessment:** Combine vegetation type with fire risk modeling.  
- **Land Cover Mapping:** Support environmental mapping and GIS-based studies.  
- **Ecological Research:** Aid biodiversity and soil conservation studies.

---

## 📊 Dataset Overview

- **Source:** [Forest Cover Type Dataset](https://drive.google.com/file/d/1kHFx_FYiu4WbP7JVm3VMPIBh4kf_fafy/view?usp=sharing)  
- **Shape:** 581,012 rows × 55 columns  
- **Target Variable:** `Cover_Type` (7 classes)

| Feature Category | Description |
|------------------|-------------|
| **Elevation, Aspect, Slope** | Terrain & topographical features |
| **Hydrology & Fire Points** | Distance to water bodies & fire ignition sources |
| **Wilderness_Area_1–4** | Binary one-hot encoded categories |
| **Soil_Type_1–40** | Binary one-hot encoded soil types |
| **Cover_Type (Target)** | 1–7 representing forest cover categories |

---

## 🔧 Project Workflow

### 1️⃣ Data Collection  
- Imported and explored dataset using Pandas.  
- Verified structure, data types, and target class distribution.

### 2️⃣ Data Understanding  
- Checked duplicates, missing values, and class imbalance.  
- Summarized statistics with `.info()`, `.describe()`, and `.value_counts()`.

### 3️⃣ Data Cleaning & Transformation  
- Handled missing values using **median imputation**.  
- Treated outliers with **Z-score / IQR methods**.  
- Fixed skewness in continuous variables using **log or sqrt transformations**.

### 4️⃣ Feature Engineering  
- Created derived columns (e.g., `hillshade_difference`, `distance_ratios`).  
- Encoded categorical variables and standardized numerical features.  

### 5️⃣ Exploratory Data Analysis (EDA)  
- Univariate & bivariate analysis using **Matplotlib** and **Seaborn**.  
- Heatmaps for correlation insights.  
- Boxplots, histograms, and class distribution visuals.  

### 6️⃣ Handling Class Imbalance  
- Applied **RandomOverSampler / SMOTE** from `imbalanced-learn` to balance training data.

### 7️⃣ Feature Selection  
- Used **Random Forest feature importance** and **correlation-based filtering** to drop redundant features.

### 8️⃣ Model Building & Evaluation  
Trained multiple classification models for comparison:  

| Model | Accuracy | Remarks |
|--------|-----------|----------|
| Decision Tree | — | Simple baseline model |
| Random Forest | — | High accuracy & robust |
| Logistic Regression | — | Low interpretability |
| KNN | — | Sensitive to scaling |
| XGBoost | — | Best performing model after tuning |

- Evaluation Metrics: **Accuracy**, **Confusion Matrix**, **Classification Report**

### 9️⃣ Hyperparameter Tuning  
- Used **GridSearchCV** / **RandomizedSearchCV** to optimize model performance.

### 🔟 Model Deployment with Streamlit  
- Built an interactive **Streamlit web app** for user input and live forest cover prediction.  
- The model predicts the **forest type (1–7)** based on user-provided environmental features.  
- Saved model as `.pkl` using `pickle` or `joblib`.

---

## 🧠 Insights & Results

- Elevation and soil type are strong predictors of forest cover type.  
- Topographic variables like slope and hillshade significantly influence classification.  
- The **XGBoost model** outperformed others after hyperparameter optimization.  
- The deployed Streamlit app enables quick, accurate cover type predictions for real-world applications.

---

## ⚙️ Tech Stack

| Category | Tools Used |
|-----------|------------|
| Language | Python |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn |
| IDE | Jupyter Notebook / VS Code |
| Web Framework | Streamlit |
| Deployment | Pickle / Joblib Model Serialization |

---

## 📦 Deliverables

- ✅ Cleaned and processed dataset  
- ✅ Comparative analysis of 5 ML models  
- ✅ Best model saved as `.pkl`  
- ✅ Streamlit web app for live predictions  
- ✅ Visualizations and feature insights  
- ✅ Documentation and report summary  

---

## 📈 Sample Visualizations

- Feature importance plot  
- Confusion matrix heatmap  
- Cover type class distribution  
- Elevation vs Cover Type boxplot  
- Correlation heatmap of top 10 features  

---

## 📎 References

- [Streamlit API Reference](https://docs.streamlit.io/develop/api-reference)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [Scikit-learn Docs](https://scikit-learn.org/stable/)  
- [Dataset Source](https://drive.google.com/file/d/1kHFx_FYiu4WbP7JVm3VMPIBh4kf_fafy/view?usp=sharing)  

---

## 🚀 Future Improvements

- Integrate geospatial visualization using **Folium / GeoPandas**  
- Deploy Streamlit app on **Render or HuggingFace Spaces**  
- Include a map-based input interface for real-time predictions  

---

## 🏷️ Tags

`Machine Learning` `Classification` `Environmental Analytics` `XGBoost` `Random Forest`  
`EDA` `Feature Engineering` `Model Deployment` `Streamlit` `Python` `Scikit-learn`

---

### 🌱 Project Outcome
**EcoType** demonstrates how **machine learning and geospatial analytics** can help predict and classify forest cover types, supporting sustainable environmental planning and forest management worldwide. 🌍🌳

