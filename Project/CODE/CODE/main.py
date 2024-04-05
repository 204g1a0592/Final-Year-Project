import pandas as pd
import seaborn
import sns as sns
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('Z:\\2021\\GROWTH PROJECTS\\By Nivesh\\TK2386---TK10272---Low Birth Weight Predictor---Dharani---CIT\\cleaned.csv')
df.head()
df.isnull().sum()
df.info()
X = df.drop(['result'],axis = 1)
y = df.result
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 20)
from sklearn.ensemble import RandomForestClassifier
dtc = DecisionTreeClassifier()
model2 = dtc.fit(x_train,y_train)
pred2 = model2.predict(x_test)
a = accuracy_score(y_test,pred2)
rfc = RandomForestClassifier()
model = rfc.fit(x_train,y_train)
pred = model.predict(x_test)
b = accuracy_score(y_test,pred)
import xgboost as xgb
xgbc = xgb.XGBClassifier()

model1 = xgbc.fit(x_train,y_train)
# print(model1.feature_importances_)
pred1 = model1.predict(x_test)
d = accuracy_score(y_test,pred1)


# print(recall_score(y_test,pred1))
# print(precision_score(y_test,pred1))

svc = svm.SVC()
model3 =svc.fit(x_train,y_train)
pred3 = model3.predict(x_test)
c = accuracy_score(y_test,pred3)

data = pd.DataFrame({'Models': ['Decision Tree', 'Random Forest', 'Support vector machine', "XGBoost"],
                     'Accuracy': [a * 100,b * 100, c * 100, d* 100]})
print(data)
ch = seaborn.barplot(x = data.Models,y= data.Accuracy)
# f = seaborn.distplot(data.Accuracy)
plt.show()
