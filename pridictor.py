import  numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../Bank_Loan_pridictor/train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('../Bank_Loan_pridictor/test_Y3wMUE5_7gLdaTN.csv')
print (train.shape, test.shape)


train.head() 
train.info() 

train.isnull().sum()

test.head()
test.info()

test.isnull().sum()

data = [train,test]
for dataset in data:
    
        categorical_columns = [x for x in dataset.dtypes.index if dataset.dtypes[x]=='object']
            
        categorical_columns = [x for x in categorical_columns if x not in ['Loan_ID' ]]
                
        for col in categorical_columns:
                        print ('\nFrequency of Categories for variable %s'%col)
                        print (train[col].value_counts())
                            
                        sns.countplot(train['Gender'])
                           
                        pd.crosstab(train.Gender, train.Loan_Status, margins = True)
                        train.Gender = train.Gender.fillna(train.Gender.mode())
                        test.Gender = test.Gender.fillna(test.Gender.mode())
gen = pd.get_dummies(train['Gender'] , drop_first = True )
train.drop(['Gender'], axis = 1 , inplace =True)
train = pd.concat([train , gen ] , axis = 1)

gen = pd.get_dummies(test['Gender'] , drop_first = True )
test.drop(['Gender'], axis = 1 , inplace =True)
test = pd.concat([test , gen ] , axis = 1)

plt.figure(figsize=(6,6))
labels = ['0' , '1', '2' , '3+']
explode = (0.05, 0, 0, 0)
size = [345 , 102 , 101 , 51]

plt.pie(size, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.show()
train.Dependents.value_counts()
       
pd.crosstab(train.Dependents , train.Loan_Status, margins = True)
train.Dependents = train.Dependents.fillna("0")
test.Dependents = test.Dependents.fillna("0")
rpl = {'0':'0', '1':'1', '2':'2', '3+':'3'}

train.Dependents = train.Dependents.replace(rpl).astype(int)
test.Dependents = test.Dependents.replace(rpl).astype(int)
sns.countplot(train['Credit_History'])
pd.crosstab(train.Credit_History , train.Loan_Status, margins = True)

train.Credit_History = train.Credit_History.fillna(train.Credit_History.mode()[0])
test.Credit_History  = test.Credit_History.fillna(test.Credit_History.mode()[0])

sns.countplot(train['Self_Employed'])

pd.crosstab(train.Self_Employed , train.Loan_Status,margins = True)

train.Self_Employed = train.Self_Employed.fillna(train.Self_Employed.mode())
test.Self_Employed = test.Self_Employed.fillna(test.Self_Employed.mode())

self_Employed = pd.get_dummies(train['Self_Employed'] ,prefix = 'employed' ,drop_first = True )
train.drop(['Self_Employed'], axis = 1 , inplace =True)
train = pd.concat([train , self_Employed ] , axis = 1)

self_Employed = pd.get_dummies(test['Self_Employed'] , prefix = 'employed' ,drop_first = True )
test.drop(['Self_Employed'], axis = 1 , inplace =True)
test = pd.concat([test , self_Employed ] , axis = 1)

sns.countplot(train.Married)

pd.crosstab(train.Married , train.Loan_Status,margins = True)

train.Married = train.Married.fillna(train.Married.mode())
test.Married = test.Married.fillna(test.Married.mode())

married = pd.get_dummies(train['Married'] , prefix = 'married',drop_first = True )
train.drop(['Married'], axis = 1 , inplace =True)
train = pd.concat([train , married ] , axis = 1)

married = pd.get_dummies(test['Married'] , prefix = 'married', drop_first = True )
test.drop(['Married'], axis = 1 , inplace =True)
test = pd.concat([test , married ] , axis = 1)

train.drop(['Loan_Amount_Term'], axis = 1 , inplace =True)
test.drop(['Loan_Amount_Term'], axis = 1 , inplace =True)

train.LoanAmount = train.LoanAmount.fillna(train.LoanAmount.mean()).astype(int)
test.LoanAmount = test.LoanAmount.fillna(test.LoanAmount.mean()).astype(int)
sns.distplot(train['LoanAmount'])

sns.countplot(train.Education)

train['Education'] = train['Education'].map( {'Graduate': 0, 'Not Graduate': 1} ).astype(int)
test['Education'] = test['Education'].map( {'Graduate': 0, 'Not Graduate': 1} ).astype(int)

sns.countplot(train.Property_Area)
train['Property_Area'] = train['Property_Area'].map( {'Urban': 0, 'Semiurban': 1 ,'Rural': 2  } ).astype(int)
test.Property_Area = test.Property_Area.fillna(test.Property_Area.mode())
test['Property_Area'] = test['Property_Area'].map( {'Urban': 0, 'Semiurban': 1 ,'Rural': 2  } ).astype(int)

sns.distplot(train['ApplicantIncome'])
sns.distplot(train['CoapplicantIncome'])

train['Loan_Status'] = train['Loan_Status'].map( {'N': 0, 'Y': 1 } ).astype(int)

train.drop(['Loan_ID'], axis = 1 , inplace =True)

train.head()
test.head()
g = sns.lmplot(x='ApplicantIncome',y='LoanAmount',data= train , col='employed_Yes', hue='Male',
          palette= ["Red" , "Blue","Yellow"] ,aspect=1.2,)
g.set(ylim=(0, 800))
plt.figure(figsize=(10,5))
sns.boxplot(x="Property_Area", y="LoanAmount", hue="Education",data=train, palette="coolwarm")
train.Credit_History.value_counts()
lc = pd.crosstab(train['Credit_History'], train['Loan_Status'])
lc.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.figure(figsize=(9,6))
sns.heatmap(train.drop('Loan_Status',axis=1).corr(), vmax=0.6, square=True, annot=True)
       
X = train.drop('Loan_Status' , axis = 1 )
y = train['Loan_Status']

X_train ,X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state =102)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train , y_train)
pred_l = logmodel.predict(X_test)
acc_l = accuracy_score(y_test , pred_l)*100
acc_l

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, y_train)
pred_rf = random_forest.predict(X_test)
acc_rf = accuracy_score(y_test , pred_rf)*100
acc_rf

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test , pred_knn)*100
acc_knn

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
pred_gb = gaussian.predict(X_test)
acc_gb = accuracy_score(y_test , pred_gb)*100
acc_gb

svc = SVC()
svc.fit(X_train, y_train)
pred_svm = svc.predict(X_test)
acc_svm = accuracy_score(y_test , pred_svm)*100
acc_svm

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
pred_gbc = gbk.predict(X_test)
acc_gbc = accuracy_score(y_test , pred_gbc)*100
acc_gbc

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forrest','K- Nearest Neighbour' ,
                 'Naive Bayes' , 'SVM','Gradient Boosting Classifier'],
                     'Score': [acc_l , acc_rf , acc_knn , acc_gb ,acc_svm ,acc_gbc ]})
models.sort_values(by='Score', ascending=False)
importances = pd.DataFrame({'Features':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False).set_index('Features')
importances.head(11) 
importances.plot.bar()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load the dataset (train file)
train = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Data preprocessing (drop rows with missing 'Loan_Status')
train = train.dropna(subset=['Loan_Status'])

# Separate features (X) and target (y)
X = train.drop(columns=['Loan_Status', 'Loan_ID'])  # Use your feature columns
y = train['Loan_Status'].map({'Y': 'Yes', 'N':'No'})  # Convert 'Y'/'N' to 1/0 for binary classification

# Handle missing values in numeric and categorical columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='median')  # Impute missing values with median for numeric columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute with the most frequent value for categorical columns

X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

# One-hot encoding for categorical features
X_encoded = pd.get_dummies(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Function to get a list of eligible candidates based on a probability threshold
def get_eligible_candidates(model, X_test, X_original, threshold=0.75):
    # Predict probabilities for the class '1' (loan approval)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Filter those with probability above the threshold
    eligible_indices = probabilities >= threshold
    
    # Get the corresponding rows from the original data
    eligible_candidates = X_original[eligible_indices].copy()  # Create a copy of eligible data
    
    # Convert eligibility from 1/0 to Yes/No in the original 'Loan_Status' column
    eligible_candidates['Loan_Status'] = ['Yes' if prob >= threshold else 'No' for prob in probabilities[eligible_indices]]
    
    return eligible_candidates

# Get the eligible candidates from the test set
X_original_test = train.iloc[X_test.index]  # Original data corresponding to test set
eligible_candidates = get_eligible_candidates(model, X_test, X_original_test)

# Save the eligible candidates to a CSV file
eligible_candidates.to_csv('eligible_candidates.csv', index=False)

print(f"Eligible candidates saved to 'eligible_candidates.csv'")





df_test = test.drop(['Loan_ID'], axis = 1)
df_test.head()
p_log = logmodel.predict(df_test)
p_rf = random_forest.predict(df_test)
predict_combine = np.zeros((df_test.shape[0]))

for i in range(0, test.shape[0]):
    temp = p_log[i] + p_rf[i]
    if temp>=2:
                predict_combine[i] = 1
                predict_combine = predict_combine.astype('int')
                submission = pd.DataFrame({
                        "Loan_ID": test["Loan_ID"],
                                "Loan_Status": predict_combine
                                    })
submission.to_csv("results.csv", encoding='utf-8', index=False)
