# Forest Fire Damage Prediction Model 🔥

A machine learning project focused on optimizing model predictions for forest fire damage extent using various strategies including outlier handling, K-fold cross validation, regularization, and ensemble methods.

## 📊 Project Overview

This project aims to predict the burned area of forest fires using meteorological and spatial data from the Montesinho Natural Park in Portugal. The dataset presents a challenging regression task due to its highly skewed target variable and complex feature relationships.

### Goal
Optimize model prediction through different strategies:
- Handling outliers
- K-fold cross validation
- Regularization techniques
- Non-linear models
- Feature engineering

## 🗂️ Dataset

**Source**: [UCI Machine Learning Repository - Forest Fires Dataset](https://archive.ics.uci.edu/dataset/162/forest+fires)

**Features**:
- **X, Y**: Spatial coordinates within the Montesinho park map (1-9, 2-9)
- **month**: Month of the year ('jan' to 'dec')
- **day**: Day of the week ('mon' to 'sun')
- **FFMC**: Fine Fuel Moisture Code index (18.7 to 96.20)
- **DMC**: Duff Moisture Code index (1.1 to 291.3)
- **DC**: Drought Code index (7.9 to 860.6)
- **ISI**: Initial Spread Index (0.0 to 56.10)
- **temp**: Temperature in Celsius (2.2 to 33.30)
- **RH**: Relative humidity in % (15.0 to 100)
- **wind**: Wind speed in km/h (0.40 to 9.40)
- **rain**: Outside rain in mm/m² (0.0 to 6.4)
- **area**: Burned area of forest in hectares (0.00 to 1090.84) - **Target Variable**

**Dataset Characteristics**:
- 517 samples
- 13 features (11 numerical, 2 categorical)
- No missing values
- Highly skewed target variable towards 0.0

## 🔧 Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## 📈 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of target variable
- Outlier detection using IQR method
- Correlation analysis between features
- Visualization of feature relationships

### 2. Feature Engineering
- **Log transformation** of target variable using `log1p()` to handle skewness
- **Cyclical encoding** for temporal features (month, day) using sine and cosine transformations
- **Outlier identification** using IQR method across all numerical features

### 3. Model Development & Optimization

#### Sequential Feature Selection
- Forward selection to identify optimal feature subsets
- Backward selection with regularized models
- Performance comparison across different feature combinations

#### Models Tested
1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 Regularization)
3. **Polynomial Features** with PCA dimensionality reduction
4. **Random Forest Regressor** (Ensemble method)

#### Cross-Validation Strategy
- 5-fold cross-validation for robust model evaluation
- Negative Mean Squared Error (MSE) as primary metric
- Root Mean Squared Error (RMSE) for interpretability

## 📊 Results

### Best Performing Models

| Model | Features | Mean RMSE | Mean MSE |
|-------|----------|-----------|----------|
| Linear Regression | temp, X | 64.05 | 4102 |
| Linear Regression | DMC | 64.13 | 4112 |
| Ridge Regression | DMC | 64.13 | 4112 |

### Key Findings
- **Best single predictor**: DMC (Duff Moisture Code)
- **Best feature combination**: temperature and X-coordinate
- **Final RMSE**: 64 hectares (within 1 standard deviation of target variable)
- **Outlier impact**: 138 outliers identified (26.7% of data)

### Model Performance Insights
- Linear models performed surprisingly well despite data complexity
- Polynomial features and PCA did not improve performance
- Random Forest showed higher variance in predictions
- Feature selection was crucial for optimal performance

## 🗺️ Spatial Analysis Observations

The analysis revealed two distinct fire-prone areas:
1. **Near-origin area** (X ≤ 3): Smaller, more manageable fires
2. **Remote forest area** (X > 3): Larger fires, potentially due to delayed detection/response


## 📝 Future Improvements

- **Ensemble Methods**: Explore more sophisticated ensemble techniques
- **Outlier Treatment**: Test model performance with outlier removal
- **Temporal Features**: Investigate seasonal patterns and trends
- **Spatial Clustering**: Implement clustering for location-based models
- **Deep Learning**: Experiment with neural networks for non-linear patterns
a star!
