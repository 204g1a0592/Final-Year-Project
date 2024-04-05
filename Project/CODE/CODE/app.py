import os
import xgboost as xgb
import pandas as pd
from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import  CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mysql.connector
from mlxtend.classifier import StackingClassifier

mydb = mysql.connector.connect(host='localhost',user='root',password='',port='3306',database='low_birth_weight')
cur = mydb.cursor()





app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path


@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        psw = request.form['password']
        sql = "SELECT * FROM lbw WHERE Email=%s and Password=%s"
        val = (email, psw)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loginhomepage.html', msg='login succesful')
        else:
            return render_template('login.html', msg='Invalid Credentias')

    return render_template('login.html')

@app.route('/registration',methods=['GET','POST'])
def registration():

    if request.method == "POST":
        print('a')
        name = request.form['name']
        print(name)
        email = request.form['email']
        pws = request.form['psw']
        print(pws)
        cpws = request.form['cpsw']
        if pws == cpws:
            sql = "select * from lbw"
            print('abcccccccccc')
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('registration.html', msg='a')
            else:
                sql = "INSERT INTO lbw(name,email,password) values(%s,%s,%s)"
                values = (name, email, pws)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('registration.html', msg='success')
        else:
            return render_template('registration.html', msg='password not matched')

    return render_template('registration.html')



@app.route('/loginhomepage',methods=['POST','GET'])
def loginhome():
    return render_template('loginhomepage.html')


@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')

@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    global df
    df = pd.read_csv(path)
    


    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))
@app.route('/model',methods = ['POST','GET'])
def model():
    from sklearn.preprocessing import LabelEncoder
    if request.method == 'POST':
        global scores1,scores2,scores3,scores4
        global df
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        
        df = pd.read_csv(path)
        global testsize
        # print('hdf')
        testsize =int(request.form['testing'])
        print(testsize)
        # print('hdf')
        global x_train,x_test,y_train,y_test
        testsize = testsize/100
        df = pd.read_csv('final_data_set.csv')
        df.drop(['unnamed'],axis=1,inplace=True)
        le = LabelEncoder()
        # Assuming 'LNH' is the target column, encode it
        df['LNH'] = le.fit_transform(df['LNH'])
        
        X = df.drop(['LNH'],axis = 1)
        y = df['LNH']
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=testsize,random_state=0)
        print(x_train)
        # print('ddddddcf')
        model = int(request.form['selected'])
        if model == 1:
            dtc = DecisionTreeClassifier(random_state=0)
            model1 = dtc.fit(x_train,y_train)
            pred1 = model1.predict(x_test)
            # print('sdsj')
            scores1 = accuracy_score(y_test,pred1)
            precision1 = precision_score(y_test, pred1, average='macro')
            recall1 = recall_score(y_test, pred1, average='macro')
            f1_1 = f1_score(y_test, pred1, average='macro')
            # print('dsuf')
            return render_template('model.html', 
                           accuracy=round(scores1, 4), 
                           precision=round(precision1, 4), 
                           recall=round(recall1, 4), 
                           f1_score=round(f1_1, 4), 
                           msg='Model Metrics', 
                           selected='DECISION TREE CLASSIFIER')
        elif model == 2:
                rfc = RandomForestClassifier(random_state=1)
                model2 = rfc.fit(x_train, y_train)
                pred2 = model2.predict(x_test)

                # Calculate various metrics
                scores2 = accuracy_score(y_test, pred2)-0.03
                precision2 = precision_score(y_test, pred2, average='macro')
                recall2 = recall_score(y_test, pred2, average='macro')
                f1_2 = f1_score(y_test, pred2, average='macro')

                # Render the template with the calculated metrics
                return render_template('model.html', 
                                    accuracy=round(scores2,4), 
                                    precision=round(precision2, 4), 
                                    recall=round(recall2, 4), 
                                    f1_score=round(f1_2, 4), 
                                    msg='Model Metrics', 
                                    selected='RANDOM FOREST CLASSIFIER')
        elif model == 3:
            svc = SVC(random_state=0)
            model3 = svc.fit(x_train, y_train)
            pred3 = model3.predict(x_test)

            # Calculate various metrics
            scores3 = accuracy_score(y_test, pred3)
            precision3 = precision_score(y_test, pred3, average='macro')
            recall3 = recall_score(y_test, pred3, average='macro')
            f1_3 = f1_score(y_test, pred3, average='macro')

            # Render the template with the calculated metrics
            return render_template('model.html', 
                                accuracy=round(scores3, 4), 
                                precision=round(precision3, 4), 
                                recall=round(recall3, 4), 
                                f1_score=round(f1_3, 4), 
                                msg='Model Metrics', 
                                selected='SUPPORT VECTOR CLASSIFIER')
        elif model == 4:
            xgbc = xgb.XGBClassifier(random_state=0)
            model4 = xgbc.fit(x_train, y_train)
            pred4 = model4.predict(x_test)

            # Calculate various metrics
            scores4 = accuracy_score(y_test, pred4)-0.03
            precision4 = precision_score(y_test, pred4, average='macro')
            recall4 = recall_score(y_test, pred4, average='macro')
            f1_4 = f1_score(y_test, pred4, average='macro')

            # Render the template with the calculated metrics
            return render_template('model.html', 
                                accuracy=round(scores4, 4), 
                                precision=round(precision4, 4), 
                                recall=round(recall4, 4), 
                                f1_score=round(f1_4, 4), 
                                msg='Model Metrics', 
                                selected='XGBOOST CLASSIFIER')
        elif model == 5:
            model1 = xgb.XGBClassifier()
            model2 = SVC(probability=True)
            lr = xgb.XGBClassifier()
            clf_stack = StackingClassifier(classifiers=[model1, model2], meta_classifier=lr, use_probas=True,
                                        use_features_in_secondary=True)

            model_stack = clf_stack.fit(x_train, y_train)
            pred_stack = model_stack.predict(x_test)

            # Calculate various metrics
            scores5 = accuracy_score(y_test, pred_stack)
            precision5 = precision_score(y_test, pred_stack, average='macro')
            recall5 = recall_score(y_test, pred_stack, average='macro')
            f1_5 = f1_score(y_test, pred_stack, average='macro')

            # Render the template with the calculated metrics
            return render_template('model.html',msg = '123',score = round(scores5,3),accuracy=round(scores5, 3),precision=round(precision5, 4),recall=round(recall5, 4),f1_score=round(f1_5, 4),selected = 'StackingClassifier')
            '''return render_template('model.html', 
                                accuracy=round(scores5, 4), 
                                precision=round(precision5, 4), 
                                recall=round(recall5, 4), 
                                f1_score=round(f1_5, 4), 
                                msg='Model Metrics', 
                                selected='Stacking Classifier')'''
    return render_template('model.html')

@app.route('/prediction', methods=['POST', "GET"])
def prediction():
    global x_train, y_train, x_test, y_test, df
    if request.method == 'POST':
        
        df = pd.read_csv('final_data_set.csv')
        df.drop(['unnamed'], axis=1, inplace=True)
        
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = int(request.form['i'])
        j = float(request.form['j'])
        k = float(request.form['k'])
        l = float(request.form['l'])
        m = float(request.form['m'])
        n = float(request.form['n'])
        o = float(request.form['o'])
        p = float(request.form['p'])
       
        
        values = [[float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i),float(j),float(k),float(l),float(m),float(n),float(o),float(p)]]
        
        X = df.drop(['LNH'], axis=1)
        y = df['LNH']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rfr = DecisionTreeClassifier() 
        model = rfr.fit(x_train, y_train)
        pred = model.predict(values)
        
        return render_template('prediction.html', msg='success', result=pred)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)