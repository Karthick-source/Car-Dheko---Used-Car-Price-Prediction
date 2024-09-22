**Car Dheko \- Used Car Price Prediction**

### **1\. Project Overview:**

This project focuses on developing a machine learning model to predict used car prices. By analyzing historical data, we aim to enhance the customer experience at CarDekho, enabling users to estimate car prices using a user-friendly Streamlit web application. The dataset includes features such as make, model, year, fuel type, transmission, kilometers driven, and more.

## **TECHNOLOGY/PACKAGES USED**

* Python 3.12  
* MySQL 8.0  
* Power Bi  
* Matplotlib  
* Seaborn  
* Abstract Syntax Tree  
* Sklearn  
* Regular Expression  
* Joblib  
* Numpy

## **SKILL-TAKEAWAY**

1. Data Cleaning and Preprocessing  
2. Exploratory Data Analysis  
3. Machine Learning Model Development  
4. Price Prediction Techniques  
5. Model Evaluation and Optimization  
6. Model Deployment  
7. Streamlit Application Development  
8. Documentation and Reporting

### **2\. Key Insights:**

* Fuel Type and Body Type seem to be significant in determining car prices. For instance, SUVs and diesel cars often command higher prices.  
* Kilometers Driven and Model Year show strong correlations with the price, indicating that newer cars with fewer kilometers are priced higher.  
* Certain manufacturers and models hold higher resale value, which is crucial for the model’s predictive accuracy.

### **3\. Methodology:**

#### **a) Data Preprocessing:**

* Removed symbols like "₹" and words like "Lakh" in the price data, converting it to a usable numerical format.  
* Cleaned categorical features like **fuel type, transmission**, and **body type**, and performed **Label encoding** to prepare them for model input.  
* Handled missing values using mean/median imputation for numerical columns and mode imputation for categorical columns.

#### **b) Exploratory Data Analysis (EDA):**

* Visualized the distribution of car prices using histograms.  
* Used scatter plots to analyze the relationships between kilometers driven, model year, and price.  
* Created box plots to examine the impact of fuel type and body type on price.

#### **c) Model Development:**

* Split the data into **80% training** and **20% testing** sets.  
* Trained multiple regression models: **Linear Regression, Decision Tree, Random Forest,** and **Gradient Boosting**.  
* Evaluated models using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)** to compare their performance.

#### **d) Model Comparison:**

* After training, the models were compared based on how accurately they predicted car prices on the test set.

**Proposed Solution**:

* Use **Random Forest** or **Gradient Boosting** as the final model for predicting used car prices.  
* Deploy this model in a **Streamlit** application where users can input car details (e.g., fuel type, model year, kilometers driven), and the app will return an estimated price.

### **5\. Challenges:**

* **Data Cleaning**: The dataset contained many missing values and inconsistencies (e.g., the price being formatted with the "₹" symbol). Cleaning these required extensive preprocessing.  
* **Feature Engineering**: Identifying which features to include in the model was critical, especially dealing with categorical variables like fuel type and body type.  
* **Outliers**: Outliers in features like kilometers driven skewed the model's predictions. Dealing with these outliers was necessary to ensure accurate model performance.  
* **Overfitting**: Some models, especially the **Decision Tree**, showed signs of overfitting during training, which was mitigated by using ensemble models like Random Forest.

### **6\. Conclusion:**

The project successfully developed a machine learning model to predict used car prices. After testing multiple algorithms, **Random Forest** and **Gradient Boosting** emerged as the top performers. These models balance predictive accuracy and generalization, making them ideal for deployment in the Streamlit application.

By implementing this solution, CarDekho can provide a tool that helps customers and sales representatives quickly estimate the value of used cars, enhancing both the buying and selling experience. The challenges in data preprocessing and model selection were overcome through systematic analysis and testing, leading to an effective and scalable solution for car price prediction.

