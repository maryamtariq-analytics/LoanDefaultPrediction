import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

#####################################################################################################
#####################################################################################################

st.title('LOAN DEFAULT PREDICTION')
st.header('*EXECUTIVE SUMMARY:*', divider='orange')
st.subheader('Business Problem:')
st.write('Loans are an important source of revenue for banks, but they are also associated with the risk that '
         'borrowers may default on their loans. A default occurs when a borrower stops making the required payments '
         'on a debt. A better estimate of the number of customers defaulting on their loan obligations will allow us to'
         ' set aside sufficient capital to absorb that loss.' )

st.subheader('Methodology:')
st.markdown("""
- Analyzed *Class Balance*
- Engineered two new variables to feed into Model as *Independent Variables*
- Modeled a *Logistic Regression Model*
- Evaluated results with various *Metrics*
""")
st.subheader('Skills:')
st.markdown("""
**Programming Language:** Python \n
**Data Manipulation Libraries:** Pandas, Numpy \n
**Machine Learning Libraries:** Scikit-Learn \n 
**Visualization Libraries:** Seaborn, Matplotlib, Plotly \n
**App and Dashboard Tool:** Streamlit \n
**Statistics and Analytical Model:** Logistic Regression
""")
st.subheader('Results:')
st.markdown("""
- As a borrower has more **Outstanding Credit Lines**, their odds of defaulting increase substantially by 3809 times.
- Borrowers with higher **Debt-to-Income Ratio** are likely to default on their loans as one-unit increase in 
*debt-to-income* will increase the odds of default by 50.4%.
- **Payment-to-Income Ratio** has only a slight impact on the probability of defaulting as one-unit increase in 
*payment-to-income ratio* will increase the odds of default by only 2%.
- Employers with longer employment histories are very much less likely to default as one-unit increase in **Years 
Employed** reduces the odds of default by about 93.5%.
- A one-unit increase in the **FICO Score** reduces the odds of defaulting by about 2.5%. While this effect is relatively 
small, it shows that a higher *FICO Score* slightly decreases the likelihood of default, which is expected as FICO scores 
are designed to assess credit risk.
""")
st.subheader('Business Recommendation:')
st.write("""
- Avoid giving loans to customers with **Outstanding Credit Lines**, **Debt-to-Income Ration** and **Payment-to-Income 
Ratio** as their chances of defaulting on loans increases.
- Customers with more **Employment Years** and higher **FICO Score** can be trusted with giving loans as chances of 
defaulting on loans are negative.
""")
st.subheader('Next Steps:')
st.write("""
- **Model Validation & Addressing Potential Over-fitting:** Perform cross-validation to ensure the model's performance 
is consistent across different data splits and not just a result of over-fitting.
- **Feature Importance Analysis:** Investigate the feature importance to understand which features contribute most to 
the model’s predictions. This can help refine the model or guide business decisions.
- **Deployment:** Start preparing for deployment (e.g., API integration, cloud deployment) and monitor it post-deployment 
for any performance drift.
- **Threshold Tuning:** Might want to adjust the classification threshold to optimize for business needs (e.g., minimize 
false positives or false negatives based on costs).
""")
######################################################################################################
######################################################################################################
url = "https://raw.githubusercontent.com/maryamtariq-analytics/LoanDefaultPrediction/refs/heads/main/Customer_Loan_Data.csv"
loan_data = pd.read_csv(url)

st.header('DATASET', divider='rainbow')
st.write('Following dataset is a sample of a *Loan Book*, and we will build a predictive *Logistic Regression Model* '
         'to estimate the *probability of loan defaults* based on customer characteristics:')
st.write(loan_data)

######################################################################################################
######################################################################################################

st.header('EXPLORATORY DATA ANALYSIS', divider='rainbow')
st.subheader('CLASS BALANCE')
# checking class balance
default = loan_data[['customer_id', 'default']]
default_total = default.groupby(['default']).count()
default_percentage = default_total / default_total.sum() * 100
default_percentage = default_percentage.rename(columns={'customer_id': 'Percentage(%)'})
st.write(default_percentage)
st.markdown('* **0:** Customers who **did not default** on their loans.')
st.markdown('* **1:** Customers who **defaulted** on their loans.')
fig0 = px.bar(default_percentage.transpose(), text_auto=True, title='CLASS BALANCE', labels={'value': 'Percentage(%)', 'index': 'customer_id'})
st.plotly_chart(fig0)
st.write('The class balance is not extremely unbalanced, so we can proceed without class rebalancing as it is roughly'
         ' near 20%-80%.')

#######################################################################################################
#######################################################################################################

st.header('FEATURE ENGINEERING', divider='rainbow')
st.write("Following new features have been created. These ratios help in assessing financial health and the manageability of"
         " debt in relation to income. Lenders use this ratios to assess a borrower’s ability to handle additional debt."
         " A higher ratio may suggest that the borrower is already highly leveraged, making them a riskier candidate for new loans.")
st.markdown("* **payment_to_income:** The ratio helps determine how much of a person’s or business's income is allocated"
            " to repaying outstanding loans.")
st.markdown("* **debt_to_income:** Financial ratio used to evaluate the proportion of an individual's or organization's"
            " income that is allocated to repaying all outstanding debts.")
# calculate 'payment to income' ratio
loan_data['payment_to_income'] = loan_data['loan_amt_outstanding'] / loan_data['income']
#calculate 'debt to income' ratio
loan_data['debt_to_income'] = loan_data['total_debt_outstanding'] / loan_data['income']
st.write(loan_data)

######################################################################################################
######################################################################################################

st.header('MODELLING', divider='rainbow')
# define variable features
x = loan_data[['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']]
y = loan_data['default']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression().fit(x_train, y_train)

y_pred = clf.predict(x_test)

code1 = '''
# Separate variables into dependent(y) and independent(x) variables 
x = loan_data[['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']]
y = loan_data['default']

# Divide the dataset into Training and Testing Data with 20% of the data in Testing Data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression Model and fit the Model to the Training Data
clf = LogisticRegression().fit(x_train, y_train)

# Make predictions on Test Data
y_pred = clf.predict(x_test)
'''
st.code(code1, language='python')

#####################################################################################################
#####################################################################################################

st.header('EVALUATION', divider='rainbow')

variable_names = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']
# Extract coefficients and intercept
coefficients = pd.DataFrame(data=clf.coef_.T, index=variable_names, columns=['Coefficient'])
intercept = clf.intercept_
# Display coefficients and intercept in a table
st.write('### COEFFICIENTS:')
st.write(coefficients)
#odd ratios
st.markdown('#### Odds-Ratios of Coefficients')
odds_ratios = np.exp(coefficients)
st.write(odds_ratios)
# intercept
st.write('### INTERCEPT:')
st.metric(label='INTERCEPT', value=f'{intercept[0]:.3f}')
# ANALYSIS
st.markdown('#### *:orange[ANALYSIS:]*')
st.markdown("""
- As a borrower has more **Outstanding Credit Lines**, their odds of defaulting increase substantially by 3809 times.
- Borrowers with higher **Debt-to-Income Ratio** are likely to default on their loans as one-unit increase in 
*debt-to-income* will increase the odds of default by 50.4%.
- **Payment-to-Income Ratio** has only a slight impact on the probability of defaulting as one-unit increase in 
*payment-to-income ratio* will increase the odds of default by only 2%.
- Employers with longer employment histories are very much less likely to default as one-unit increase in **Years 
Employed** reduces the odds of default by about 93.5%.
- A one-unit increase in the **FICO Score** reduces the odds of defaulting by about 2.5%. While this effect is relatively 
small, it shows that a higher *FICO Score* slightly decreases the likelihood of default, which is expected as FICO scores 
are designed to assess credit risk.
""")

######################################################################################################

st.subheader('CONFUSION MATRIX')
# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# Create Matplotlib figure
fig, ax = plt.subplots()
# Plot confusion matrix
disp.plot(ax=ax, cmap=plt.cm.Blues)
# Customize plot
plt.title('Confusion Matrix')
plt.grid(False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Display the plot in Streamlit
st.pyplot(fig)
# ANALYSIS
st.markdown('#### *:orange[ANALYSIS:]*')
st.write("""
- **True Negatives:** Values predicted as *Not a Defaulter* and they actually did not default in their debt payments.
1649 values are predicted as True Negatives.  
- **True Positives:** Values predicted as *Loan Defaulter* and they actually are defaulters. 338 values are predicted as 
True Positives.
- **False Positives:** Values predicted as *Loan Defaulter* but in reality they did not default. Only 3 values have been 
predicted as False Positives.  
- **False Negatives:** Values predicted as *Not a Defaulter* but they did default. Only 10 values are predicted as False
Negatives.
""")
######################################################################################################

st.subheader('METRICS')
# Calculate metric scores
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
roc_auc_score = metrics.roc_auc_score(y_test,y_pred)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(label='PRECISION SCORE', value=format(precision, '.3f'))
col2.metric(label='RECALL SCORE', value=format(recall, '.3f'))
col3.metric(label='ACCURACY SCORE', value=format(accuracy, '.3f'))
col4.metric(label='F1 SCORE', value=format(f1, '.3f'))
col5.metric(label='ROC AUC SCORE', value=format(roc_auc_score, '.3f'))

# create roc curve display object
disp1 = RocCurveDisplay.from_predictions(y_test, y_pred)
# plot roc curve via matplotlib
fig1, ax = plt.subplots()
disp1.plot(ax=ax)
plt.title('ROC Curve')
plt.legend(loc='lower right')
# display in streamlit
st.pyplot(fig1)
# ANALYSIS
st.markdown('#### *:orange[ANALYSIS:]*')
st.markdown("""
- **PRECISION SCORE:** A precision score of 0.991 means that ***99.1% of the instances classified as positive are truly 
positive***. This indicates very few false positives, meaning the model is highly accurate when it predicts a positive 
class. The model makes very few mistakes when predicting the positive class (low false positives).
- **RECALL SCORE:** A recall score of 0.971 means that ***97.1% of all actual positive cases were correctly identified*** 
by the model. The model is excellent at identifying true positives, but it may have a small number of false negatives 
(about 2.9% of actual positives are missed).
- **ACCURACY SCORE:** An accuracy score of 0.994 means that ***99.4% of all predictions made by the model are correct***, 
including both positive and negative classes. The model is very reliable, with a high rate of correct predictions for 
both classes.
- **F1 SCORE:** An F1 score of 0.981 indicates a ***good balance between precision and recall***. The model has an 
excellent balance between correctly identifying positives and avoiding false positives.
- **ROC AUC SCORE:** A ROC AUC score of 0.985 means that the model ***performs extremely well at distinguishing between 
positive and negative cases***. The model is highly effective at distinguishing between classes, with an excellent ability 
to separate positives from negatives.
""")
st.write('Metric Scores are nearing 100% which tells that our Logistic Regression Model is successful in rightly '
         'classifying data between *Loan Defaulter* and *Not a Defaulter*.')

###################################################################################################
###################################################################################################

st.header('BUSINESS RECOMMENDATIONS', divider='rainbow')
st.write("""
- Avoid giving loans to customers with **Outstanding Credit Lines**, **Debt-to-Income Ration** and **Payment-to-Income 
Ratio** as their chances of defaulting on loans increases.
- Customers with more **Employment Years** and higher **FICO Score** can be trusted with giving loans as chances of 
defaulting on loans are negative.
""")

####################################################################################################
####################################################################################################

st.header('NEXT STEPS', divider='rainbow')
st.write("""
- **Model Validation & Addressing Potential Over-fitting:** Perform cross-validation to ensure the model's performance 
is consistent across different data splits and not just a result of over-fitting.
- **Feature Importance Analysis:** Investigate the feature importance to understand which features contribute most to 
the model’s predictions. This can help refine the model or guide business decisions.
- **Deployment:** Start preparing for deployment (e.g., API integration, cloud deployment) and monitor it post-deployment 
for any performance drift.
- **Threshold Tuning:** Might want to adjust the classification threshold to optimize for business needs (e.g., minimize 
false positives or false negatives based on costs).
""")

####################################################################################################
####################################################################################################

st.divider()