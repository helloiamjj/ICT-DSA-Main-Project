#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries




### Data Wrangling

import pandas as pd
import numpy as np

### Data Visualization

import seaborn as sns
import matplotlib.pyplot as plt


### Remove unnecessary warnings

import warnings
warnings.filterwarnings('ignore')


#### Importing the data




# load train data
train_df = pd.read_csv("train.csv",low_memory=False)
# Load test data
test_df = pd.read_csv("test.csv")







# Re-size the dataset

train_df = train_df[:8000] 
test_df = test_df[:4000]



### Shape of the dataset

print(train_df.shape)
print(test_df.shape)


# The dataset consists of 28 columns and 100000 rows.


# Check for null values

print(train_df.isna().sum())
print("\n\n")
print(test_df.isna().sum())





# From the above data,the columns - Age, Annual_Income, Num_of_Loan, Num_of_Delayed_Payment, Changed_Credit_Limit, Outstanding_Debt, Amount_invested_monthly, Monthly_Balance are object data type so we will change the datatype of these columns from object to a numerical datatype like int or float.



### Summary statistics of the numerical columns in the dataset

train_df.describe()





# Identify numerical columns with symbols present in rows

train_df.select_dtypes(include='object').columns




# select those 8 columns

convert_cols = ['Age',
       'Annual_Income', 'Num_of_Loan',
       'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Outstanding_Debt', 
       'Amount_invested_monthly','Monthly_Balance']






#Remove unwanted characters from the selected columns using for loop
#removes - and _ from beginning and end of values

for col in convert_cols:  
    train_df[col] = train_df[col].str.strip('-_') 
    test_df[col] = test_df[col].str.strip('-_')




# remove unwanted characters from rest of the columns
# replace empty row with nan
# to apply the changes to whole dataframe

train_df.replace('', np.NaN, inplace=True) 
train_df.replace('_______', np.NaN, inplace=True)

test_df.replace('', np.NaN, inplace=True) 
test_df.replace('_______', np.NaN, inplace=True)


#### 2. Convert the datatype to int/ float for above columns




train_df['Age']=train_df['Age'].astype(int)
train_df['Annual_Income']=train_df['Annual_Income'].astype(float)
train_df['Num_of_Loan']=train_df['Num_of_Loan'].astype(int)
train_df['Num_of_Delayed_Payment']=train_df['Num_of_Delayed_Payment'].astype(float)
train_df['Changed_Credit_Limit']=train_df['Changed_Credit_Limit'].astype(float)
train_df['Outstanding_Debt'] = train_df['Outstanding_Debt'].astype(float)
train_df['Amount_invested_monthly']=train_df['Amount_invested_monthly'].astype(float)
train_df['Monthly_Balance'] =train_df['Monthly_Balance'].astype(float)




test_df['Age']=test_df['Age'].astype(int)
test_df['Annual_Income']=test_df['Annual_Income'].astype(float)
test_df['Num_of_Loan']=test_df['Num_of_Loan'].astype(int)
test_df['Num_of_Delayed_Payment']=test_df['Num_of_Delayed_Payment'].astype(float)
test_df['Changed_Credit_Limit']=test_df['Changed_Credit_Limit'].astype(float)
test_df['Outstanding_Debt'] = test_df['Outstanding_Debt'].astype(float)
test_df['Amount_invested_monthly']=test_df['Amount_invested_monthly'].astype(float)
test_df['Monthly_Balance'] =test_df['Monthly_Balance'].astype(float)



# can drop ID, Name and SSN as they are identifiers and not useful for visualization

train_df1 = train_df.drop(['ID', 'Name', 'SSN'], axis=1)
test_df1 = test_df.drop(['ID', 'Name', 'SSN'], axis=1)






### Replace error value with nan

train_df1['Credit_Mix'] = train_df1['Credit_Mix'].replace('_', np.nan)
test_df1['Credit_Mix'] = test_df1['Credit_Mix'].replace('_', np.nan)





# replace error value with nan

train_df1['Payment_Behaviour'] = train_df1['Payment_Behaviour'].replace('!@9#%8', np.nan)
test_df1['Payment_Behaviour'] = test_df1['Payment_Behaviour'].replace('!@9#%8', np.nan)




# ### Univariate analysis


# #Occupation
# train_df1['Occupation'].value_counts().plot.pie(autopct='%.1f%%')
# plt.title("Distribution of Occupations")
# plt.show() 




# sns.histplot(train_df1['Monthly_Inhand_Salary'], color='darkgreen')
# plt.title('Distribution of Monthly Inhand Salary')
# plt.show()


# # The inhand salary distribution is right-skewed. Mean > median. Most of the customers have inhand salary in the range of 1000-5000 dollars.



# sns.histplot(train_df1['Delay_from_due_date'], kde=True,color='orangered')
# plt.title('Distribution of Delay from Due date')
# plt.show()





# # Majority of customers did payment after due date.Only few paid before the due date (-ve values). 
# # KDE (Kernel Density Estimate) - A smooth curve showing continuous probability density.Maximum count at 14.



# sns.countplot(x= train_df1['Credit_Mix'], palette="YlOrBr") 
# #different types of credit accounts of a customer, shows the ability to handle multiple credits
# plt.title("Distribution of Credit Mixes")
# plt.show()


# # Majority of credit mix type is standard.


# sns.histplot(train_df1['Outstanding_Debt'], kde=True, color='orangered')# debt left to be paid
# plt.title('Distribution of Outstanding Debt')
# plt.show()


# # The distribution is skewed to the right. Most people have less amount of debt left to be paid.

# # In[38]:


# sns.countplot(x= train_df1['Credit_Score'],palette="YlOrBr")
# plt.title('Distribution of Credit Scores')
# plt.show()


# # There is an uneven distribution of target column values.

# # ### Bivariate analysis (w.r.t. target)


# # Annual_Income vs credit score
# sns.barplot(x=train_df1['Annual_Income'], y=train_df1['Credit_Score'])
# plt.title('Annual Income vs Credit Score')
# plt.show()


# # Customers with higher annual income tend to have better credit scores.



# # Monthly_Inhand_Salary vs credit scores
# sns.boxplot(y=train_df1['Credit_Score'], x=train_df1['Monthly_Inhand_Salary'])
# plt.title('Monthly Inhand Salary vs Credit Score')
# plt.show()


# # The distribution is right skewed(mean>median) for all three credit scorers.Customers with higher monthly inhand salary are showing better credit scores.




# # Delay_from_due_date vs credit score
# sns.barplot(y=train_df1['Credit_Score'], x=train_df1['Delay_from_due_date'])
# plt.title('Delay from due date vs Credit Score')
# plt.show()


# # Credit scores get poorer as delay from due date increases.



# # Credit_Mix vs Credit scores
# cat_var = pd.crosstab(train_df1.Credit_Mix, train_df1.Credit_Score)
# print(cat_var)
# sns.heatmap(cat_var, annot=True,fmt='.4g',cmap='Blues')
# plt.title('Credit Mix vs Credit Score')
# plt.show()


# # Customers with better credit mix, shows better credit scores.


# #Credit_Utilization_Ratio vs Credit score
# sns.boxplot(y=train_df1['Credit_Utilization_Ratio'], x=train_df1['Credit_Score'])
# plt.title('Credit Utilization Ratio vs Credit Score')
# plt.show()


# # Credit Utilization ratio distribution is almost even for good and standard credit scorers.But, the range is slightly lower for poor category.Up to a range increase in Credit Utilization ratio is good for better credit score. Beyond limit, credit utilization negatively affects credit score.


# ### Distribution of Payment_of_Min_Amount for each Credit Score

# sns.catplot(x= 'Payment_of_Min_Amount', col= 'Credit_Score', data = train_df1, kind = 'count', col_wrap = 3)
# plt.show()


# # Most of the Customers with poor and standard credit scores did only the minimum payment. Most of the customers with good credits scores, did more than the minimum payment.From the above graphs, we can see that the most of the customers with a good credit score didn't pay the minimum amount for the loan. 



# #Total_EMI_per_month vs credit scores
# sns.barplot(x=train_df1['Total_EMI_per_month'], y=train_df1['Credit_Score'])
# plt.title('Total EMI per month vs Credit Score')
# plt.show()


# Customers that paid higher EMI tend to have better credit scores.



# change the default number of rows to be displayed 

pd.set_option('display.max_columns', None)




# ## Handling missing values, Outliers





# freqgraph = train_df1.select_dtypes(include = ['float'])
# freqgraph.hist(figsize =(20,15))
# plt.show()





#  drop, Num_Bank_Accounts, Type_of_Loan as they do not contribute to target

train_df1.drop(['Num_Bank_Accounts','Type_of_Loan'],axis =1 , inplace = True )
test_df1.drop(['Num_Bank_Accounts','Type_of_Loan'],axis =1 , inplace = True )




# Filling the null values in Occupation, Monthly_Inhand_Salary,Credit_Mix using mode. 

train_df1['Occupation'] = train_df1.groupby('Customer_ID')['Occupation'].apply(lambda x: x.fillna(x.mode()[0]))

train_df1['Monthly_Inhand_Salary'] = train_df1.groupby('Customer_ID')['Monthly_Inhand_Salary'].apply(lambda x: x.fillna(x.mode()[0]))

train_df1['Credit_Mix'] = train_df1.groupby('Customer_ID')['Credit_Mix'].apply(lambda x: x.fillna(x.mode()[0]))

## fill test data
test_df1['Occupation'] = test_df1.groupby('Customer_ID')['Occupation'].apply(lambda x: x.fillna(x.mode()[0]))

test_df1['Monthly_Inhand_Salary'] = test_df1.groupby('Customer_ID')['Monthly_Inhand_Salary'].apply(lambda x: x.fillna(x.mode()[0]))

test_df1['Credit_Mix'] = train_df1.groupby('Customer_ID')['Credit_Mix'].apply(lambda x: x.fillna(x.mode()[0]))





# fill using forward fill ,as no. of delayed payments depend on previous month value

train_df1['Num_of_Delayed_Payment'] = train_df1['Num_of_Delayed_Payment'].fillna(method='ffill')
test_df1['Num_of_Delayed_Payment'] = test_df1['Num_of_Delayed_Payment'].fillna(method='ffill')



# changed credit limit

train_df1[['Customer_ID','Changed_Credit_Limit']].head(24)




# replace missing values using groupby mode

train_df1['Changed_Credit_Limit'] = train_df1.groupby('Customer_ID')['Changed_Credit_Limit'].apply(lambda x: x.fillna(x.mode()[0]))
test_df1['Changed_Credit_Limit'] = test_df1.groupby('Customer_ID')['Changed_Credit_Limit'].apply(lambda x: x.fillna(x.mode()[0]))




# changed Num_Credit_Inquiries

train_df1[['Customer_ID','Num_Credit_Inquiries']].head(24)



# replace missing values using groupby mode

train_df1['Num_Credit_Inquiries'] = train_df1.groupby('Customer_ID')['Num_Credit_Inquiries'].apply(lambda x: x.fillna(x.mode()[0]))
test_df1['Num_Credit_Inquiries'] = test_df1.groupby('Customer_ID')['Num_Credit_Inquiries'].apply(lambda x: x.fillna(x.mode()[0]))






# replace NM with nan
train_df1.loc[train_df1['Payment_of_Min_Amount']=='NM','Payment_of_Min_Amount']=np.nan
test_df1.loc[test_df1['Payment_of_Min_Amount']=='NM','Payment_of_Min_Amount']=np.nan



# fill nan using mode
train_df1['Payment_of_Min_Amount'] = train_df1.groupby('Customer_ID')['Payment_of_Min_Amount'].apply(lambda x: x.fillna(x.mode()[0]))
test_df1['Payment_of_Min_Amount'] = test_df1.groupby('Customer_ID')['Payment_of_Min_Amount'].apply(lambda x: x.fillna(x.mode()[0]))




# changed Amount_invested_monthly

train_df1[train_df1['Amount_invested_monthly'].isna()==True]



# handle outliers before filling null values
# replace 10,000 with nan 

train_df1['Amount_invested_monthly'] = train_df1['Amount_invested_monthly'].replace(10000, np.nan)
test_df1['Amount_invested_monthly'] = test_df1['Amount_invested_monthly'].replace(10000, np.nan)




#  fill all null values using median 

train_df1['Amount_invested_monthly'] = train_df1.groupby('Customer_ID')['Amount_invested_monthly'].apply(lambda x: x.fillna(x.median()))
test_df1['Amount_invested_monthly'] = test_df1.groupby('Customer_ID')['Amount_invested_monthly'].apply(lambda x: x.fillna(x.median()))



# groupby and fill payment behaviour using mode

train_df1['Payment_Behaviour'] = train_df1.groupby('Customer_ID')['Payment_Behaviour'].apply(lambda x: x.fillna(x.mode()[0]))
test_df1['Payment_Behaviour'] = test_df1.groupby('Customer_ID')['Payment_Behaviour'].apply(lambda x: x.fillna(x.mode()[0]))




# find the extreme value of Monthly_Balance

train_df1[train_df1['Monthly_Balance']== max(train_df1['Monthly_Balance'])] 




# replace the extreme value of Monthly_Balance with nan

train_df1['Monthly_Balance'] = train_df1['Monthly_Balance'].replace(max(train_df1['Monthly_Balance']), np.nan)
test_df1['Monthly_Balance'] = test_df1['Monthly_Balance'].replace(max(test_df1['Monthly_Balance']), np.nan)




# use median to fill nan values

train_df1['Monthly_Balance'] = train_df1.groupby('Customer_ID')['Monthly_Balance'].apply(lambda x: x.fillna(x.median()))
test_df1['Monthly_Balance'] = test_df1.groupby('Customer_ID')['Monthly_Balance'].apply(lambda x: x.fillna(x.median()))


# ### Feature Engineering



train_df1['Credit_History_Age']



# to extract numeric part of Credit History age column
def convert_to_month(x):
    if pd.notnull(x):
        num1 = int(x.split(' ')[0])
        num2 = int(x.split(' ')[3])
        #print(num1, num2)
        return (num1*12)+num2
    else:
        return x




# convert age to months
train_df1['Credit_History_Age'] = train_df1.Credit_History_Age.apply(lambda x: convert_to_month(x)).astype(float)
test_df1['Credit_History_Age'] = test_df1.Credit_History_Age.apply(lambda x: convert_to_month(x)).astype(float)




# fill the missing values in Credit History Age

train_df1['Credit_History_Age'] = train_df1.groupby('Customer_ID')['Credit_History_Age'].apply(lambda x: x.fillna(x.mode()[0]))
test_df1['Credit_History_Age'] = test_df1.groupby('Customer_ID')['Credit_History_Age'].apply(lambda x: x.fillna(x.mode()[0]))




# We have successfully removed null values from train dataset and test dataset.




#Box plot of numerical columns

# num_col = train_df1.select_dtypes(include=np.number).columns.tolist()

# plt.figure(figsize=(20,30))

# for i, variable in enumerate(num_col):
#                      plt.subplot(5,4,i+1)
#                      plt.boxplot(train_df1[variable],whis=1.5)
#                      plt.tight_layout()
#                      plt.title(variable)



num_col = test_df1.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(20,30))

for i, variable in enumerate(num_col):
                     plt.subplot(5,4,i+1)
                     plt.boxplot(train_df1[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)





# drop the other columns

num_col1=train_df1.drop(['Credit_Score','Credit_History_Age','Payment_of_Min_Amount','Payment_Behaviour','Credit_Mix','Customer_ID','Month','Occupation','Delay_from_due_date','Changed_Credit_Limit','Monthly_Balance','Amount_invested_monthly','Credit_Utilization_Ratio'], axis = 1)







# Identify the outliers and remove 

for i in num_col1:
    Q1=train_df1[i].quantile(0.25) # 25th quantile
    Q3=train_df1[i].quantile(0.75) # 75th quantile
    IQR = Q3-Q1
    Lower_Whisker = Q1 - 1.5*IQR 
    Upper_Whisker = Q3 + 1.5*IQR
    train_df1[i] = np.clip(train_df1[i], Lower_Whisker, Upper_Whisker)




# PLot the numerical columns
# plt.figure(figsize=(20,30))
# for i, variable in enumerate(num_col):
#                      plt.subplot(5,4,i+1)
#                      plt.boxplot(train_df1[variable],whis=1.5)
#                      plt.tight_layout()
#                      plt.title(variable)


# ### ENCODING

# Categorical columns:
#     
# - Customer_ID -drop column
# - Month - Label encode
# - Occupation - Label encode
# - Credit_Mix - One Hot encode
# - Payment_of_Min_Amount - One Hot encode
# - Payment_Behaviour - One Hot Encode
# - Credit_Score - map target as 0,1,2


# drop Customer ID
train_df1 = train_df1.drop('Customer_ID', axis=1)
test_df1 = test_df1.drop('Customer_ID', axis=1)

# drop payment behaviour
train_df1 = train_df1.drop('Payment_Behaviour', axis=1)
test_df1 = test_df1.drop('Payment_Behaviour', axis=1)

# #### ONE HOT ENCODING


replace_map = {'Payment_of_Min_Amount': {'Yes': 1, 'No': 0} }

train_df1.replace(replace_map, inplace=True)
test_df1.replace(replace_map, inplace=True)




# #### LABEL ENCODING



#import library
from sklearn.preprocessing import LabelEncoder



#fit the model
label_encoder=LabelEncoder()



for i in ['Month', 'Occupation']:  
    train_df1[i]=label_encoder.fit_transform(train_df1[i])
    le_name_mapping =dict((zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print(le_name_mapping)



for i in ['Month', 'Occupation']:  
    test_df1[i]=label_encoder.fit_transform(test_df1[i])
    le_name_mapping =dict((zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print(le_name_mapping)


train_df1 = train_df1.drop('Credit_Mix', axis=1)
test_df1 = test_df1.drop('Credit_Mix', axis=1)
# Mapping Credit score

replace_map = {'Credit_Score': {'Poor': 0, 'Good': 2, 'Standard': 1 }}




train_df1.replace(replace_map, inplace=True)




test_df1.replace(replace_map, inplace=True)





# ### SCALING

# <!-- MIN MAX SCALING -->




# ### STANDARD SCALING



# scale_cols = ['Age',  'Annual_Income', 'Monthly_Inhand_Salary',
#        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
#        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
#        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio','Credit_History_Age',
#        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

# from sklearn.preprocessing import StandardScaler

# std = StandardScaler()

# train_df1[scale_cols]= std.fit_transform(train_df1[scale_cols])



# test_df1[scale_cols] = std.fit_transform(test_df1[scale_cols])




train_df1.describe()



test_df1.describe()



X = train_df1.drop('Credit_Score', axis=1)
y= train_df1['Credit_Score']






# ### PCA

# PCA will help us to find a reduced number of features that will represent our original dataset in a compressed way, capturing up to a certain portion of its variance depending on the number of new features we end up selecting.



# from sklearn.decomposition import PCA



# pca = PCA(n_components=0.97)
# X_pca = pca.fit_transform(X)



# X_pca.shape





# test_pca=pca.transform(test_df1)




# test_pca.shape



# print(pca.explained_variance_ratio_)



# len(pca.explained_variance_ratio_)



# split data into test and train
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.25, random_state=42, stratify=y)




print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))





# from sklearn.linear_model import LogisticRegression as lgrClassifier 
# from sklearn.multiclass import OneVsRestClassifier
# # Creating a Logistic Regression model

# lgr = lgrClassifier(C = 100)
# o_vs_r = OneVsRestClassifier(lgr)
# o_vs_r.fit(X_train , y_train)
# lgr_pred = o_vs_r.predict(X_test)
# lgr_scoret = o_vs_r.score(X_test, y_test)




# # predict using test df
# lgr_pred_t = o_vs_r.predict(test_pca)




# lgr_pred_t[:25]




from sklearn.metrics import classification_report



# print(classification_report(y_test, lgr_pred))





# from sklearn.neighbors import KNeighborsClassifier as knnClassifier
# from sklearn.metrics import accuracy_score

# #find optimum number of neighbors
# k_range = range(1, 15)
# k_scores = []

# for k in k_range:
#     knn = knnClassifier(n_neighbors=k)
#     o_vs_r = OneVsRestClassifier(knn)
#     o_vs_r.fit(X_train , y_train)
#     scores = accuracy_score(y_test, o_vs_r.predict(X_test))
#     k_scores.append(scores)

# # Plot the results
# plt.plot(k_range, k_scores)
# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy scores')
# plt.grid()
# plt.show()


# # k-value 8 is showing highest accuracy.




# knn = knnClassifier(n_neighbors=8)
# o_vs_r = OneVsRestClassifier(knn)
# o_vs_r.fit(X_train , y_train)
# knn_pred = o_vs_r.predict(X_test)
# knn_scoret = o_vs_r.score(X_test , y_test)




# print(classification_report(y_test, knn_pred))




# knn_pred_t = o_vs_r.predict(test_pca)




# knn_pred_t[:25]




# from sklearn.svm import SVC


# svm_clf = SVC(kernel = 'linear')
# o_vs_r = OneVsRestClassifier(svm_clf)
# o_vs_r.fit(X_train, y_train)
# svm_pred = o_vs_r.predict(X_test)
# svm_scoret = o_vs_r.score(X_test , y_test)


# # In[122]:


# print(classification_report(y_test, svm_pred))


# # In[123]:


# svm_pred_t = o_vs_r.predict(test_pca)


# # In[124]:


# svm_pred_t[0:25]


# # ### 4) Decision Tree
# # 
# # The intuition behind Decision Trees is that you use the dataset features to create yes/no questions and continually split the dataset until you isolate all data points belonging to each class.
# # 
# # With this process you’re organizing the data in a tree structure.Every time you ask a question you’re adding a node to the tree. And the first node is called the root node.The result of asking a question splits the dataset based on the value of a feature, and creates new nodes.
# # If you decide to stop the process after a split, the last nodes created are called leaf nodes.
# # 
# # ![image.png](attachment:image.png)

# # In[125]:



# from sklearn.tree import DecisionTreeClassifier


# # In[126]:


# # modeling 

# dt = DecisionTreeClassifier(random_state=42,max_depth=10,min_samples_split=2)
# dt.fit(X_train, y_train)
# dt_pred = dt.predict(X_test)

# dt_scoret = dt.score(X_test, y_test)


# # In[127]:


# print(classification_report(y_test, dt_pred))


# # In[128]:


# dt_pred_t = dt.predict(test_pca)


# # In[129]:


# dt_pred_t[0:25]


# # ### 5) Random Forest

# # It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model. Random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
# # 
# # ![image.png](attachment:image.png)

# # In[130]:


# from sklearn.ensemble import RandomForestClassifier as rfClassifier

# rf = rfClassifier(max_features= 14, max_depth= 8)
# rf.fit(X_train , y_train)
# rf_pred = rf.predict(X_test)
# rf_scoret = rf.score(X_test , y_test)


# # In[131]:


# print(classification_report(y_test, rf_pred))


# # In[132]:


# rf_pred_t = rf.predict(test_pca)


# # In[133]:


# rf_pred_t[:25]


# # ### 6) Extreme Gradient Boosting (XGBoost)
# # ![image.png](attachment:image.png)
# # 
# # In XGBoost, a tree ensemble model is constructed by training many decision trees on subsets of the data, using a process called boosting. The decision trees are trained in a sequential manner, with each tree being trained to correct the mistakes of the previous tree.

# # In[134]:


# import xgboost as xgb


# # In[135]:


# # modeling 
# xgboost = xgb.XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=100, num_class=3)
# xgboost.fit(X_train, y_train)

# xgb_pred = xgboost.predict(X_test)
# xgb_scoret = xgboost.score(X_test, y_test)


# # In[136]:


# print(classification_report(y_test, xgb_pred))


# # In[137]:


# xgb_pred_t = xgboost.predict(test_pca)


# # In[138]:


# xgb_pred_t[:25]


# # ### 7) Light gbm 

# # Light GBM is a fast, distributed, high-performance gradient boosting framework that uses a tree-based learning algorithm.
# # Light GBM splits the tree leaf-wise with the best fit whereas other boosting algorithms split the tree depth-wise or level-wise rather than leaf-wise. In other words, Light GBM grows trees vertically while other algorithms grow trees horizontally.
# # 

# # In[139]:


# import lightgbm as lgb
# from lightgbm import LGBMClassifier


# # In[140]:


# lgb_model = LGBMClassifier(num_class=3)
# lgb_model.fit(X_train, y_train)
 
# # Predicting the Target variable
# lgb_pred = lgb_model.predict(X_test)
# lgb_scoret = lgb_model.score(X_test, y_test)


# # In[141]:


# print(classification_report(y_test, lgb_pred))


# # In[142]:


# lgb_pred_t =lgb_model.predict(test_pca)


# # In[143]:


# lgb_pred_t[:25]





from catboost import CatBoostClassifier


# In[145]:


# modeling 
catboost = CatBoostClassifier(random_state=42,classes_count=3, verbose=False)
catboost.fit(X_train, y_train)
cat_pred = catboost.predict(X_test)
cat_scoret = catboost.score(X_test, y_test)


# In[146]:


print(classification_report(y_test, cat_pred))


# In[147]:


cat_pred_t = catboost.predict(test_df1)


# In[148]:


cat_pred_t[:25].reshape(1,-1)


# In[149]:


# # evaluate the models
# models = [ 'Logistic Regression','KNN','SVM-linear','Decision Tree','Random Forest', 'XGBoost','Light gbm','CatBoost']
# data = [ lgr_scoret*100,knn_scoret*100, svm_scoret*100,dt_scoret*100,rf_scoret*100, xgb_scoret*100,lgb_scoret*100, cat_scoret*100]
# cols = ['Accuracy Score']
# pd.DataFrame(data=data , index= models , columns= cols).sort_values(by=['Accuracy Score'], ascending= False)


# # In[150]:


# #plot the accuracies
# plt.plot(models,data,'o-')
# plt.xlabel("Models")
# plt.ylabel('Accuracy scores')
# plt.legend(cols)
# plt.xticks(rotation=90)
# plt.grid()
# plt.show()


# Catboost is showing best accuracy (75.75%).

# ### Stratified KFold

# In K-Fold cross-validation, the data is divided into K folds, and the model is trained and evaluated K times, using a different fold as the test set in each iteration. Stratified K-Fold cross-validation is similar, except that it ensures that the proportion of the target class is the same in each fold as it is in the original dataset.
# 
# 

# In[151]:


# ### Stratified KFold
# from sklearn.model_selection import StratifiedKFold, cross_val_score

# # Create a list of models to evaluate
# model_names = [ 'Logistic Regression','KNN','SVM-linear','Decision Tree','Random Forest', 'XGBoost','Light gbm','CatBoost']
# models = [lgr, knn, svm_clf, dt, rf, xgboost,lgb_model,catboost]
# mean_scores = []

# # Split the data into K folds, ensuring that the distribution of classes is the same in each fold
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Use a for loop to iterate through the models
# for model in models:
#     # Calculate the cross-validated accuracy using the `cross_val_score` function
#     scores = cross_val_score(model, X_pca, y, cv=kfold, scoring='accuracy')
#     mean_scores.append(scores.mean())

#     # Print the mean and standard deviation of the scores for the current model
#     print(f'{model.__class__.__name__}: {scores.mean():.2f} ')


# # In[152]:


# #plot the accuracies
# plt.plot(model_names,mean_scores,'o-')
# plt.xlabel('Models')
# plt.ylabel("Accuracies")
# plt.xticks(rotation=90)
# plt.grid()
# plt.show()


# Catboost is showing best accuracy(75%).

# ## Fine-tuning

# ### Logistic regression model

# In[153]:


# from sklearn.model_selection import GridSearchCV

# # Create a logistic regression model
# lgr = lgrClassifier(class_weight='balanced')
# o_vs_r = OneVsRestClassifier(lgr)

# # Define the hyperparameter 'C'-regularization strength
# param_grid = {'estimator__C': [0.1, 1, 10, 100, 1000]}

# # Use grid search to find the best hyperparameters
# grid_search = GridSearchCV(o_vs_r, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Use the best model to make predictions on the test set
# best_model = grid_search.best_estimator_
# lgr_pred = best_model.predict(X_test)
# lgr_scoret = best_model.score(X_test, y_test)


# # In[154]:


# print(f"Best model:{best_model},accuracy is :{lgr_scoret*100:.2f}%")


# # ### kNN 

# # In[155]:


# #define the hyperparameters- distance metric and weights
# param_grid = {'estimator__n_neighbors': [3, 5, 7,8, 11,14],
#               'estimator__weights': ['uniform', 'distance'],
#               'estimator__p': [1, 2]}

# # Create the KNN classifier
# knn = knnClassifier()

# # Create the OneVsRestClassifier with the KNN classifier as the estimator
# o_vs_r = OneVsRestClassifier(knn)

# # Set up the grid search
# grid_search = GridSearchCV(o_vs_r, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# knn_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{knn_scoret*100:.2f}%")


# # ### SVM

# # In[156]:


# #define the hyperparameters-regularization strength, kernel
# param_grid = {'estimator__C': [0.1, 1, 10],
#               'estimator__gamma': [0.01, 0.1, 1, 10],
#               'estimator__kernel': ['linear']}

# # Create the SVM classifier
# svm_clf = SVC()
# o_vs_r = OneVsRestClassifier(svm_clf)

# # Set up the grid search
# grid_search = GridSearchCV(o_vs_r, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# svm_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{svm_scoret*100:.2f}%")


# # ### Decision Tree

# # In[157]:


# # Define the hyperparameters 
# param_grid = {'max_depth': [2, 4, 6, 8],
#               'max_leaf_nodes': [2, 4, 6, 8]}

# # Create the decision tree classifier
# dt = DecisionTreeClassifier()

# # Set up the grid search
# grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# best_model = grid_search.best_estimator_
# dt_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{dt_scoret*100:.2f}%")


# # ### Random Forest

# # In[159]:


# # Define the hyperparameters
# param_grid = {'n_estimators': [10, 50, 100],
#               'max_depth': [2, 4, 6, 8],
#               'min_samples_leaf': [1, 2, 4]}

# # Create the random forest classifier
# rf = rfClassifier()

# # Set up the grid search
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# # Print the best hyperparameters
# best_model = grid_search.best_estimator_
# rf_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{rf_scoret*100:.2f}%")


# # ### XGBoost

# # In[168]:


# # Define the hyperparameters
# param_grid = {'learning_rate': [0.1, 0.5, 1],
#               'max_depth': [2, 4, 6, 8]}

# # Create the XGBoost classifier
# xgboost = xgb.XGBClassifier(num_class=3)

# # Set up the grid search
# grid_search = GridSearchCV(xgboost, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# best_model = grid_search.best_estimator_
# xgb_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{xgb_scoret*100:.2f}%")


# # ### LightGBM

# # In[169]:


# #Define the hyperparameters
# param_grid = {'learning_rate': [0.01, 0.1, 1],
#               'n_estimators': [50, 100, 200],
#               'num_leaves': [5, 10, 20]}

# # Create the LightGBM classifier
# lgb_model = LGBMClassifier(num_class=3)

# # Set up the grid search
# grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# # Print the best hyperparameters
# best_model = grid_search.best_estimator_
# lgb_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{lgb_scoret*100:.2f}%")


# # ### CatBoost

# # In[176]:


# # Define the hyperparameters
# param_grid = {'depth': [3, 6, 9],'n_estimators':[100, 200, 300]}

# # Create the CatBoost classifier
# catboost = CatBoostClassifier(verbose=False)

# # Set up the grid search
# grid_search = GridSearchCV(catboost, param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# best_model = grid_search.best_estimator_
# cat_scoret = best_model.score(X_test , y_test)
# print(f"Best model:{best_model},accuracy is :{cat_scoret*100:.2f}%")

# Serialize the python object using pickle
import pickle
pickle.dump(catboost, open('catboost.pkl', 'wb'))

print(catboost.predict(X_test))