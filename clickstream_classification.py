from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import pandas as pd
from Encoding import DataEncoding
import mlflow
class classification:
  def __init__(self):
    pass

  def logistic_regression(self,x_train,x_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    acc=accuracy_score(y_test,prediction)
    f1=f1_score(y_test,prediction)
    precision=precision_score(y_test,prediction)
    recall=recall_score(y_test,prediction)
    roc_auc=roc_auc_score(y_test,prediction)
    mlflow.set_experiment("clickclassification")
    with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1 score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall score", recall)
            mlflow.log_metric("roc auc", roc_auc)
            mlflow.sklearn.log_model(model, "logistic_regression")

    print('Accuracy= ',acc)
    print('F1=',f1)
    print('Precision=',precision)
    print('Recall=',recall)
    print('ROC AUC=',roc_auc)
    return model, acc

  def decision_tree(self,x_train,x_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    acc=accuracy_score(y_test,prediction)
    f1=f1_score(y_test,prediction)
    precision=precision_score(y_test,prediction)
    recall=recall_score(y_test,prediction)
    roc_auc=roc_auc_score(y_test,prediction)
    
    mlflow.set_experiment("clickclassification")
    with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1 score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall score", recall)
            mlflow.log_metric("roc auc", roc_auc)
            mlflow.sklearn.log_model(model, "decision tree")

    print('Accuracy= ',acc)
    print('F1=',f1)
    print('Precision=',precision)
    print('Recall=',recall)
    print('ROC AUC=',roc_auc)
    return model, acc

  def random_forest(self,x_train,x_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    acc=accuracy_score(y_test,prediction)
    f1=f1_score(y_test,prediction)
    precision=precision_score(y_test,prediction)
    recall=recall_score(y_test,prediction)
    roc_auc=roc_auc_score(y_test,prediction)

    
    mlflow.set_experiment("clickclassification")
    with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1 score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall score", recall)
            mlflow.log_metric("roc auc", roc_auc)
            mlflow.sklearn.log_model(model, "random_forest")

    print('Accuracy= ',acc)
    print('F1=',f1)
    print('Precision=',precision)
    print('Recall=',recall)
    print('ROC AUC=',roc_auc)
    
    return model, acc

  
  def evaluate_models(self, x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": self.logistic_regression,
        "Decision Tree": self.decision_tree,
        "Random Forest": self.random_forest,
        
    }
    best_model = None
    best_acc = float('-inf')  # Fix this line
    best_model_name = ""

    for name, model_func in models.items():
        trained_model, acc = model_func(x_train, x_test, y_train, y_test)  # Fix missing variable
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_model_name = name  

    print('Best Model: ', best_model_name)
    print('Best Accuracy: ', best_acc)

    with open(f"{best_model_name}_clickstream.pkl", "wb") as f:
        pkl.dump(best_model, f)


