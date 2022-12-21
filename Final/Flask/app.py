from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

df = pd.read_csv("diabetes.csv")
columns = ['BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction']
df = df.replace(0, pd.np.nan).dropna(subset=columns).fillna(0)

quant_cols_1 = df.drop('Outcome', axis=1)
Outcome = df['Outcome']

x_cols = df.columns[:(len(df.columns)-1)]
x_train, x_test, y_train, y_test = train_test_split(quant_cols_1, Outcome, test_size=0.3, random_state = 42)
#scaler=StandardScaler()
X=df.drop(columns=["Outcome"])
#X=scaler.fit_transform(X)
y = df["Outcome"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 42)



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index1.html")


@app.route("/data", methods=["GET"])
def data():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    a = request.args.get("preg")
    b = request.args.get("glu")
    c = request.args.get("bp")
    d = request.args.get("st")
    e = request.args.get("i")
    f = request.args.get("bmi")
    g = request.args.get("dpf")
    h = request.args.get("age")
    X_test1 = [[a,b,c,d,e,f,g,h]]
    #X_test1 = [[6,148,72,35,0,33.6,0.627,50]]
    #X_test2 = [[2,141,58,34,128,25.4,0.699,24]]
    logistic_reg = LogisticRegression(solver='liblinear')
    logistic_reg.fit(X_train, y_train)
    pred = logistic_reg.predict(X_test1)
    if pred[0] == 1.0:
        return render_template("prediction2.html")
    elif pred[0] == 0.0:
        return render_template("prediction.html")
    return
@app.route("/info", methods=["GET"])
def info():
    usertext = request.args.get("models")
    if usertext == "K-Nearest Neighbors Prediction Model":
        k_val = int(request.args.get("K-Nearest"))
        knn = KNeighborsClassifier(k_val)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        report = report.round(3)

        m = confusion_matrix(y_test, prediction)
        ul=m[0][0]
        ur=m[0][1]
        ll=m[1][0]
        lr=m[1][1]
        accuracy_test = round((ul+lr)/(ul+lr+ll+ur), 3)

        prediction_trained = knn.predict(X_train)
        report_train = pd.DataFrame(classification_report(y_train, prediction_trained, output_dict=True))
        report_train = report_train.round(3)
        m_1 = confusion_matrix(y_train, prediction_trained)
        ul_1=m_1[0][0]
        ur_1=m_1[0][1]
        ll_1=m_1[1][0]
        lr_1=m_1[1][1]
        accuracy_trained = round((ul_1+lr_1)/(ul_1+lr_1+ll_1+ur_1), 3)
        return render_template("knn.html",k_val=k_val,report_train=report_train,report=report, accuracy_test = accuracy_test, accuracy_trained = accuracy_trained, ul=ul, ur=ur,ll=ll, lr=lr, ul_1=ul_1, ur_1=ur_1,ll_1=ll_1, lr_1=lr_1)
    
    elif usertext == "Random Forest Prediction Model":
        random_forest = RandomForestClassifier(n_estimators=200)
        random_forest.fit(X_train, y_train)
        prediction = random_forest.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        report = report.round(3)
        
        m = confusion_matrix(y_test, prediction)
        ul=m[0][0]
        ur=m[0][1]
        ll=m[1][0]
        lr=m[1][1]
        accuracy_test = round((ul+lr)/(ul+lr+ll+ur), 3)

        prediction_trained = random_forest.predict(X_train)
        report_train = pd.DataFrame(classification_report(y_train, prediction_trained, output_dict=True))
        report_train = report_train.round(3)
        m_1 = confusion_matrix(y_train, prediction_trained)
        ul_1=m_1[0][0]
        ur_1=m_1[0][1]
        ll_1=m_1[1][0]
        lr_1=m_1[1][1]
        accuracy_trained = round((ul_1+lr_1)/(ul_1+lr_1+ll_1+ur_1), 3)
        return render_template("random_forest.html",report_train=report_train,report=report, accuracy_test = accuracy_test, accuracy_trained = accuracy_trained, ul=ul, ur=ur,ll=ll, lr=lr, ul_1=ul_1, ur_1=ur_1,ll_1=ll_1, lr_1=lr_1)
   
    elif usertext == "Logistic Regression Prediction Model":
        logistic_reg = LogisticRegression(solver='liblinear')
        logistic_reg.fit(X_train, y_train)
        prediction = logistic_reg.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        report = report.round(3)

        m = confusion_matrix(y_test, prediction)
        ul=m[0][0]
        ur=m[0][1]
        ll=m[1][0]
        lr=m[1][1]
        accuracy_test = round((ul+lr)/(ul+lr+ll+ur), 3)

        prediction_trained = logistic_reg.predict(X_train)
        report_train = pd.DataFrame(classification_report(y_train, prediction_trained, output_dict=True))
        report_train = report_train.round(3)
        m_1 = confusion_matrix(y_train, prediction_trained)
        ul_1=m_1[0][0]
        ur_1=m_1[0][1]
        ll_1=m_1[1][0]
        lr_1=m_1[1][1]
        accuracy_trained = round((ul_1+lr_1)/(ul_1+lr_1+ll_1+ur_1), 3)
        return render_template("logistic_reg.html",report_train=report_train,report=report, accuracy_test = accuracy_test, accuracy_trained = accuracy_trained, ul=ul, ur=ur,ll=ll, lr=lr, ul_1=ul_1, ur_1=ur_1,ll_1=ll_1, lr_1=lr_1)
    
    elif usertext == "XGBoost Prediction Model":
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        xgb_model.fit(X_train, y_train)
        prediction = xgb_model.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        report = report.round(3)

        m = confusion_matrix(y_test, prediction)
        ul=m[0][0]
        ur=m[0][1]
        ll=m[1][0]
        lr=m[1][1]
        accuracy_test = round((ul+lr)/(ul+lr+ll+ur), 3)

        prediction_trained = xgb_model.predict(X_train)
        report_train = pd.DataFrame(classification_report(y_train, prediction_trained, output_dict=True))
        report_train = report_train.round(3)
        m_1 = confusion_matrix(y_train, prediction_trained)
        ul_1=m_1[0][0]
        ur_1=m_1[0][1]
        ll_1=m_1[1][0]
        lr_1=m_1[1][1]
        accuracy_trained = round((ul_1+lr_1)/(ul_1+lr_1+ll_1+ur_1), 3)
        return render_template("xgboost.html",report_train=report_train,report=report, accuracy_test = accuracy_test, accuracy_trained = accuracy_trained, ul=ul, ur=ur,ll=ll, lr=lr, ul_1=ul_1, ur_1=ur_1,ll_1=ll_1, lr_1=lr_1)
    return
    

if __name__ == "__main__":
    app.run(debug=True)