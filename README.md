# HR Data Analysis Project
`Pritpal-Singh for Heliverse`

## Introduction
This project involves the analysis of HR employee attrition data. It includes various data exploration, preprocessing, and visualization tasks to understand the relationships within the dataset.<br>
This code is designed for data exploration and preprocessing tasks using Python's popular libraries such as Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Scikit-learn. It involves loading a dataset, exploring its structure, handling missing values, and performing data type conversions. <br>


## Code Overview
The given code performs the following 3 major steps(so given in the assignment)
- Dataset Analysis and Preprocessing
- Model Development
- Model Evaluation and Optimization



### Step 1: Install Required Libraries
Ensure that you have the following Python libraries installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
from warnings import filterwarnings

filterwarnings("ignore")
```

You can install these libraries using pip



### Step 2: Download and Import the Dataset
Download the dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv` from your source or repository and place it in the same directory as your Python script.

```python
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df = pd.DataFrame(data)
df
```

### Step 3: Data Discovery

#### Exploratory Data Analysis
- Categorical and numerical columns are separated to facilitate further analysis.
- Univariate analysis is performed on categorical data to visualize the distribution of each category.
- Categorical data is visualized against the target variable "Attrition" to understand its impact.
- Numerical data is visualized using boxplots to identify outliers and understand the distribution of values.
#### Preprocessing Steps
- Categorical attributes with only two distinct values are converted to numeric using label encoding.
- Categorical attributes with more than two distinct values are converted to numeric using one-hot encoding.

![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/132cb572-7ecf-4d6a-a318-7c22310d47a9)



### Step 4: Discovering Relationship in Data

#### Visualizing Correlations
- Correlation between variables is visualized using a bar plot to understand their relationship with the target variable "Attrition."
- Correlation between numerical variables is visualized using a heatmap with Spearman correlation coefficient.
  
#### Univariate analysis of categorical data:
```python
sns.set(rc={"axes.facecolor":"white","figure.facecolor":"#9ed9cd"})
sns.set_palette("pastel")
for i, col in enumerate(cat):

    fig, axes = plt.subplots(1,2,figsize=(10,5))

    # count of col (countplot)
    
    ax=sns.countplot(data=df, x=col, ax=axes[0])
    activities = [var for var in df[col].value_counts().sort_index().index]
    ax.set_xticklabels(activities,rotation=90)
    for container in axes[0].containers:
        axes[0].bar_label(container)
        
    # count of col (pie chart)
    
    index = df[col].value_counts().index
    size = df[col].value_counts().values
    explode = (0.05, 0.05)

    axes[1].pie(size, labels=index,autopct='%1.1f%%', pctdistance=0.85)

    # Inner circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.suptitle(col,backgroundcolor='black',color='white',fontsize=15)

    plt.show()
```
![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/5363f4e8-2c63-490b-97a9-3a78a6c64bc0)

![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/3282c08d-1a9e-4549-8e2f-b949d4d62a31)

![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/27893a11-dd70-4c7a-8e7a-ff33e8eee9bb)



### Step 5: Developing a Machine Leraning Model

#### Normalization
- Min-max scaling is applied to normalize the feature values between 0 and 1.
  
```python
#normalizing 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
DF1 = DF.drop(columns=['Attrition'])
norm = scaler.fit_transform(DF)
norm_df = pd.DataFrame(norm,columns=DF.columns)
```
#### Oversampling
- SMOTE (Synthetic Minority Over-sampling Technique) is utilized to handle class imbalance by generating synthetic samples for the minority class.

``` python
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(x_train,y_train) 
```

#### Model Training and Evaluation

1. Logistic Regression model is trained on the oversampled data.
```python
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(C=1000,max_iter=10000)
log_reg.fit(smote_train, smote_target)
y_pred_lg = log_reg.predict(x_test)

print ('accuracy',metrics.accuracy_score(y_test, y_pred_lg))
```
- Model performance is evaluated using accuracy and confusion Follow these
![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/1f10bd9c-c098-4110-a5b8-27f85e3284a8)



2. Random Forest

```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier()
rfc = rfc.fit(smote_train , smote_target)
y_pred = rfc.predict(x_test)

print ('accuracy',metrics.accuracy_score(y_test, y_pred))

```

- Model performance is evaluated using accuracy and confusion Follow these 
![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/7b0f474c-8f7d-406c-81e7-75ad176a4dfe)


3. Gradient Boost

```python

seed=0
gb_params ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0}

from sklearn.ensemble import  GradientBoostingClassifier

gb = GradientBoostingClassifier(**gb_params)
gb.fit(smote_train, smote_target)

gb_predictions = gb.predict(x_test)
print('accuracy',metrics.accuracy_score(y_test, gb_predictions))
```

- Model performance is evaluated using accuracy and confusion Follow these
![image](https://github.com/pritpalcodes/Heliverse---Pritpal-SIngh/assets/90276050/5bd6e8e8-282c-4886-a7c2-d164ac092426)
