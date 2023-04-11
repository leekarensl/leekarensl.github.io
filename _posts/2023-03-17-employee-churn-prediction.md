---
layout: post
title: Predicting Employee Churn Using ML
image: "/posts/employeechurn-title-img.png"
tags: [Employee Churn Prediction, HR Analytics, Machine Learning, Classification, Python]
---

Our client, company based in London, is concerned about retaining their high performing employees and wants to utilise Machine Learning to predict exactly which of its employees are most at risk of leaving.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Exploratory Analysis](#exploratory-title)
- [04. Random Forest](#rf-title)
- [05. Application](#modelling-application)
- [06. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a company based in London is concerned that their high performing employees are leaving them and have asked us to investigate how retention of such employees can be improved. They have provided us with a dataset containing attributes of its current and past employees and have requested for us to develop a model to predict which of their employees are most at risk of leaving and to understand the drivers behind it. Due to the amount of time spent interviewing and finding replacements for lost employees, the true cost of replacing an employee can often be quite large. Understanding why and when employees are likely to leave can help the company review their recruitment and retention strategy.

We will use Machine Learning for this specfic task.

<br>
<br>
### Actions <a name="overview-actions"></a>

With the dataset provided by the client, we found that 84% of employees actually stayed with the company while only 16% left. This is a highly unbalanced dataset but it is also not unusual in most business cases. As we will be predicting a binary output, we will use Random Forest for our classification modelling approach. Random Forest is partcularly good at making predictions if you have a lot of data with many features. It is a robust model using the prediction of many decision trees. Each decision tree looks at a different subset of the data and features and makes its own prediction. The Random Forest algorithm combines all the predictions by taking the majority vote. In addition, Feature Importance from Random Forest will also allow the client to understand the key drivers behind employees leaving. 

As the dataset is highly unbalanced, we will also ensure that we do not rely on the classification accuracy alone when assessing the results. We will also be analysing Precision, Recall and the F1-Score. We will import the data but will need to pre-process the data based on the requirement for the Random Forest algorithm. We will then train and test the model and then measure the predictive performance using Classification Accuracy, Precision, Recall and F1 scores.

<br>
<br>

### Results <a name="overview-results"></a>

Random Forest was chosen as the model due to its robustness and that it will allow the client to understand the key drivers behind employees leaving. It's predictive performance are summarised as follows:

<br>
**Metric 1**

Classification Accuracy = 0.893

<br>
**Metric 2**

Precision = 0.884

<br>
**Metric 3**

Recall = 0.906

<br>
**Metric 4**

F1 Score = 0.895

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected (e.g. reasons for job satisfaction scores, reasons for leaving etc.), and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting employee churn.
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the binary *Left* metric from the dataset provided by the client. The table below shows the column names of the dataset.

<br>
<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| Left | Dependent | The variable showing if the employee left the company or not, categorised as Yes or No|
| Gender | Independent | The gender of the employee, categorised as Male or Female |
| MonthlyIncome | Independent | The monthly income of the employee, categorised as low, medium or high |
| Department | Independent | The department the employee belonged to |
| NumCompaniesWorked | Independent | The total number companies the employee has worked in |
| Over18 | Independent | If the employee is over 18, categorised as Y or N |
| workingfromhome | Independent | If the employee has the option of working from home or not, categorised as Yes or No |
| BusinessTravel | Independent | How frequently the employee travels for work, categorised as Travel_Rarely, Travel_Frequently or Non_Travel |
| DistanceFromHome | Independent | The distance in miles the employee lives from their work place |
| StandardHours | Independent | The number of working hours per week |
| JobSatisfaction | Independent | The job satisfaction score of the employee, with 1 as the lowest and 4 as the highest score |
| complaintfiled | Independent | If the employee has ever filed a complaint with the company |
| PercentSalaryHike | Independent | The most recent percentage salary increase of the employee |
| PerformanceRating | Independent | The employee's most recent performance rating, with 1 as the lowest and 5 as the highest score |
| TotalWorkingYears | Independent | The employee's total number of working experience |
| YearsAtCompany | Independent | The employee's tenure in years in the company |
| YearsSinceLastPromotion | Independent | The number of years lapsed sinced the employee was last promoted |

<br>
# Modelling Overview  <a name="modelling-overview"></a>

We will build a model that looks to accurately predict *Left*, based upon the employee data listed above.

If that can be achieved, we can use this model to predict the probability of future employee leaving, allowing the company to act, minimising the chance of the company losing a good employee. 

As we are predicting a binary output and due to the unbalanced nature of the dataset, we will be using the Random Forest algorithm.

<br>

# Initial Exploratory Analysis <a name="exploratory-title"></a>

### Data Import <a name="ea-import"></a>

We will be importing the dataset which was in a csv format. As 'Over18' and 'StandardHours' are 'Y' and '40' respectively for all the rows in the dataset, they are not useful for analysis or for building the machine learning model. So let's drop both. We will also need to import any libraries needed:

```python
import pandas as pd

# importing data for initial analysis
df_hr = pd.read_csv("employee.csv")

# dropping both columns
df_hr = df_hr.drop(["Over18","StandardHours"], axis = 'columns')

```
The dataset contained some categorical variables which we will deal with using One Hot encoding later on. For now, let's just look at the numerical features and see if there are any correlation to the target variable of employee leaving. We will first have to apply one hot encoding to the target variable 'Left' as this is itself a categorical variable:

```python
df_hr = pd.get_dummies(df_hr, columns = ['Left'],drop_first = True)

# doing correlations with categorical variables in it produces warning messages.The following code simply ignores it.
import warnings
warnings.filterwarnings("ignore")

# setting 'Left_Yes' to be the target and the dropping it from the df_hr dataframe
df_hr['target'] = df_hr['Left_Yes']
df_hr = df_hr.drop(["Left_Yes"], axis = 'columns')
correlations = df_hr.corr()['target'].sort_values()
print('Positive correlations: \n', correlations.tail(8))
print('\nNegative correlations: \n', correlations.head(11))

```

![correlation](/img/posts/correlation-entire.png "Correlations with Left_Yes")

It appears that 'Total Working Years', 'Age', 'Years At Company' and 'Job Satisfaction' has the highest correlation with employee leaving. As the client is concerned with high performing employees leaving, let's also investigate the data from their high performing employees only and see how it compares with the entire dataset. We have defined high performing employees as having a **PerformanceRating** score of 3 or greater.

<br>

The steps above were repeated for high performing employees data only and the results can be summarised as follows:

![correlation-compare](/img/posts/correlation-high-vs-rest1.png "High vs Rest")

Although the strengths of the correlations differ slightly when looking at the data for high employees only, the Top 4 variables explaining employee attrition remains similar. The factors impacting employee attrition can be summarised as:
- Job Experience (Total Working Years and Years At Company)
- Job Satisfaction <br>

Age has a high positive correlation with Total Working Years (0.68) and Years At Company (0.31). Regardless of high or poor performers, the peak of employees leaving appears to be around the late 20s - early 30s group.<br>

![age](/img/posts/age-chart.png "Age High vs Low")

Let's bear the above factors in mind when we look at building the machine learning model later on. For now, let's just now look at each of the factor:

### Job Experience

<br>

![experience-compare](/img/posts/job-experience.png "Job Experience")

It appears that employees were more likely to leave the company at the early stage of their career. The peak of employee attrition happened on the frst year with high performers more likely to leave. This may be caused by employees figuring out their options at the early career stage or felt that there was a mismatch of the job role with their expectations. The attrition trend continues to fluctuate but stablizes around the 10th year. <br>

### Job Satisfaction

<br>

![satisfaction-compare](/img/posts/job-satisfaction.png "Job Satisfaction")

There is a higher number of high performing employees leaving when their job satisfaction score was 1 than with any other scores.


<br>

# Random Forest <a name="rf-title"></a>

Let's now look at building the prediction model. We will utlise the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="rf-import"></a>

For simplicity, let's just re-import the dataset into a data_for_model dataframe and drop the 2 variables 'Over18' and 'StandardHours' as they will not be useful in the machine learning model.  We will also ensure that our dataset is being shuffled. In addition we will also investigate the class balance of our dependent variable.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTETomek

# import modelling data
data_for_model = pd.read_csv("employee.csv")

# drop uneccessary columns
data_for_model.drop("Over18", "StadardHours" axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

Unlike other classification models like Logistic Regression, a benefit of using Random Forest is that it is not susceptible to the effects of outliers or highly correlated input variables. However as there are a number of categorical independent variables in this dataset, these will need to be pre-processed. As an example, one of the categorical variables in the dataset is Gender where values are 'Male' or 'Female'. The Random Forest algorithm can't deal with data in this format as it can't assign any numerical meaning to it when assessing the relationship between the Gender independent variable and the dependent variable. As gender doesn't have any explicit order to it, in other words, Male isn't higher or lower than Female and vice versa, one approach is to apply One Hot Encoding to this and all other categorical columns.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of new columns for each categorical variable with either a 1 or a 0 saying whether that value is true or not for that observation. These new columns would go into our model as input variables and the original column is discarded.

We also drop one of the new columns using the parameter drop_first = True. We do this to avoid the dummy variable trap where our newly created encoded columns perfectly predict each other.


```python

# One hot encoding for all categorical variables
data_for_model = pd.get_dummies(data_for_model, columns = ['Gender'],drop_first = True)
data_for_model = pd.get_dummies(data_for_model, columns = ['MonthlyIncome'],drop_first = True)
data_for_model = pd.get_dummies(data_for_model, columns = ['Department'],drop_first = True)
data_for_model = pd.get_dummies(data_for_model, columns = ['BusinessTravel'],drop_first = True)
data_for_model = pd.get_dummies(data_for_model, columns = ['complaintfiled'],drop_first = True)
data_for_model = pd.get_dummies(data_for_model, columns = ['Left'],drop_first = True)

```
We will also investigate the class balance of our dependent variable. This is important when assessing classification accuracy.

```python
# Checking balance of dataset
data_for_model["Left_Yes"].value_counts(normalize = True)

```

<br>
##### SMOTETomek Sampling and Splitting Data For Modelling
The dataset is found to be imbalanced with 84% of employees staying and only 16% choosing to leave the company. When a dataset is imbalanced, it will lead to bias during model training with the class containing a higher number of samples (in this case, No) preferred more over the class containing a lower number of samples (Yes). In order to overcome this bias of the model, we need to make the dataset balanced, containing an approximately equal number of samples in both classes.

One way of achieving this is using the SMOTETomek sampling method once the data has been split out for modelling. SMOTETomek works by combining two methods; SMOTE and Tomek links. SMOTE creates new synthetic examples of the minority class by interpolating between existing examples. Tomek links identify pairs of examples from the different classes that are very close to each other and they are removed from the dataset.

<br>
```python
# Splitting data into X and y objects for modelling
X = data_for_model.drop(["Left_Yes"], axis = 1)
y = data_for_model["Left_Yes"]

# Using SMOTETomek to sample to create a balanced dataset
smk = SMOTETomek()
X,y=smk.fit_resample(X,y)
X.shape,y.shape

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

```
In the code above we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable. We then use the SMOTETomek sampling method to create a more balanced dataset.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. We use the stratify parameter to ensure that both our training and test sets have the same proportion of employees who stayed in the company or left, meaning that we can be more confident in our assessment of the model's predictive performance.

<br>

### Model Training <a name="rf-model-training"></a>

Instantiating and training our Random Forest model is done using the below code.  We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

We also look to build more Decision Trees in the Random Forest (500) than would be done using the default value of 100.

Lastly, since the default scikit-learn implementation of Random Forests does not limit the number of randomly selected variables offered up for splitting at each split point in each Decision Tree - we put this in place using the *max_features* parameter.  This can always be refined later through testing, or through an approach such as gridsearch.

```python

# instantiate our model object
clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 5)

# fit our model using our training & test sets
clf.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

We need to assess how well our model is predicting on new data - we use the trained model object (here called *clf*) and ask it to predict the *Left* variable for the test set.

In the code below we create one object to hold the binary 1/0 predictions, and another to hold the actual prediction probabilities for the positive class.

```python

# predict on the test set
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

```

<br>
##### Confusion Matrix

A Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations.

The below code creates and plots the Confusion Matrix using the *ConfusionMatrixDisplay* functionality from within scikit-learn a

```python

# creates and plots the confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_class, cmap = "coolwarm")
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.show()

```

<br>
![alt text](/img/posts/rf1-confusion-matrix.png "Random Forest Confusion Matrix")

<br>
The aim is to have a high proportion of observations falling into the top left cell (predicted non-leavers and actual non-leavers) and the bottom right cell (predicted leavers and actual leavers).

Since the proportion of leavers in our data was around 16:84 we will analyse the model accuracy not only using Classification Accuracy, but also Precision, Recall, and F1-Score as they will help us assess how well our model has performed from different points of view.

<br>
#### Classification Performance Metrics
<br>
**Classification Accuracy**

Classification Accuracy is a metric that tells us *of all predicted observations, what proportion did we correctly classify*.  This is very intuitive, but when dealing with imbalanced classes, can be misleading.  

An example of this could be a rare disease. A model with a 98% Classification Accuracy on might appear like a fantastic result, but if our data contained 98% of patients *without* the disease, and 2% *with* the disease - then a 98% Classification Accuracy could be obtained simply by predicting that *no one* has the disease - which wouldn't be a great model in the real world.

In this example of the rare disease, we could define Classification Accuracy as *of all predicted patients, what proportion did we correctly classify as either having the disease, or not having the disease*

<br>
**Precision & Recall**

Precision is a metric that tells us *of all observations that were predicted as positive, how many actually were positive*

Keeping with the rare disease example, Precision would tell us *of all patients we predicted to have the disease, how many actually did*

Recall is a metric that tells us *of all positive observations, how many did we predict as positive*

Again, referring to the rare disease example, Recall would tell us *of all patients who actually had the disease, how many did we correctly predict*

The tricky thing about Precision & Recall is that it is impossible to optimise both - it's a zero-sum game.  If you try to increase Precision, Recall decreases, and vice versa.  Sometimes however it will make more sense to try and elevate one of them, in spite of the other.  In the case of our rare-disease prediction like we've used in our example, perhaps it would be more important to optimise for Recall as we want to classify as many positive cases as possible.  In saying this however, we don't want to just classify every patient as having the disease, as that isn't a great outcome either!

So - there is one more metric which is actually a *combination* of both.

<br>
**F1 Score**

F1-Score is a metric that essentially "combines" both Precision & Recall.  Technically speaking, it is the harmonic mean of these two metrics.  A good, or high, F1-Score comes when there is a balance between Precision & Recall, rather than a disparity between them.

Overall, optimising your model for F1-Score means that you'll get a model that is working well for both positive & negative classifications rather than skewed towards one or the other.  To return to the rare disease predictions, a high F1-Score would mean we've got a good balance between successfully predicting the disease when it's present, and not predicting cases where it's not present.

Using all of these metrics in combination gives a really good overview of the performance of a classification model, and gives us an understanding of the different scenarios & considerations!

In the code below, we utilise in-built functionality from scikit-learn to calculate these four metrics.

```python

# classification accuracy
accuracy_score(y_test, y_pred_class)

# precision
precision_score(y_test, y_pred_class)

# recall
recall_score(y_test, y_pred_class)

# f1-score
f1_score(y_test, y_pred_class)

```
<br>
Running this code gives us:

* Classification Accuracy = **0.893** meaning we correctly predicted the class for 89.3% of the test set observations
* Precision = **0.884** meaning that for our *predicted* leavers, we were correct 88.4% of the time
* Recall = **0.905** meaning that of all *actual* leavers, we predicted correctly 90.5% of the time
* F1-Score = **0.895**


<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.  

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest, we can measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomised?*

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

One way of doing this is called **Feature Importance**. This is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the gini impurity score (for a Classification problem) was before the split was made, and compare this to the gini impurity score after the split was made.  We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

<br>
```python

# calculate feature importance
feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

```
<br>
That code gives us the plot as seen below:

<br>
![alt text](/img/posts/rf2-classification-feature-importance.png "Random Forest Feature Importance Plot")

<br>
According to the Random Forest algorithim, it appears that Age, Job Satisfaction and Job Experience (Total Working Years, Years at Company) are the top drivers in explaining employee churn. This matches our earlier findings!

___
<br>
# Application <a name="modelling-application"></a>

We now have our findings as well as our model object, and the required pre-processing steps to use this model for the next time the company receives new employee data.  When this is ready to launch we can feed the neccessary employee information, obtaining predicted probabilities for each employee leaving. We will also share our insights from the dataset with the client for them to take action. Perhaps communication on role expectations or company culture can be made clearer at the start of the recruitment funnel to reduce employee attrition on the first year. Further data collection will also be useful in understanding and improving employees' job satisfaction score. Some examples include but not limiting to job autonomy, workload, relationship with manager, development opportunities etc.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting employee churn.
