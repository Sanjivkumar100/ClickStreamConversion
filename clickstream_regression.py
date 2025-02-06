import pickle as pkl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error,mean_squared_error
from Encoding import DataEncoding
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import mlflow

class Regression:
    def __init__(self):
        pass  

    def linear_regression(self, x_train, x_test, y_train, y_test):
        model = LinearRegression()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        r2 = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)
        rmse=root_mean_squared_error(y_test,prediction)
        mse=mean_squared_error(y_test,prediction)
        print(f"Linear Regression - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
        mlflow.set_experiment("clickstreamregression")
        with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "linear_regression")
        return model, r2, mae
    

    def lasso_regression(self, x_train, x_test, y_train, y_test):
        model = Lasso(alpha=1.0)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        r2 = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)
        rmse=root_mean_squared_error(y_test,prediction)
        mse=mean_squared_error(y_test,prediction)
        print(f"Lasso Regression - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
        mlflow.set_experiment("clickstreamregression")
        with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "lasso_regression")
        return model, r2, mae

    def ridge_regression(self, x_train, x_test, y_train, y_test):
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        r2 = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)
        rmse=root_mean_squared_error(y_test,prediction)
        mse=mean_squared_error(y_test,prediction)
        print(f"Ridge Regression - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
        mlflow.set_experiment("clickstreamregression")
        with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "ridge_regression")
        return model, r2, mae

    def gradient_boosting_regressor(self, x_train, x_test, y_train, y_test):
       parameters=dict(n_estimators=1000,max_depth=5,learning_rate=0.1,random_state=42)
       gbr_full=GradientBoostingRegressor(**parameters)
       gbr_early_stopping=GradientBoostingRegressor(**parameters,validation_fraction=0.1,n_iter_no_change=10)
       gbr_full.fit(x_train,y_train)
       n_estimators_full = gbr_full.n_estimators_
       gbr_early_stopping.fit(x_train, y_train)
       estimators_early_stopping = gbr_early_stopping.n_estimators_
       optimal_n_estimators = gbr_early_stopping.n_estimators_
       final_model = GradientBoostingRegressor(
            n_estimators=optimal_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,)
       Xfulltrain=x_train
       yfulltrain=y_train
       final_model.fit(Xfulltrain,yfulltrain)
       y_pred_gbr=final_model.predict(x_test)
       mae=mean_absolute_error(y_test,y_pred_gbr)
       r2=r2_score(y_test,y_pred_gbr)
       rmse=root_mean_squared_error(y_test,y_pred_gbr)
       mse=mean_squared_error(y_test,y_pred_gbr)
       print(f"Gradient Boosting - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
       mlflow.set_experiment("clickstreamregression")
       with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(final_model, "gradient_regression")
       return final_model, r2, mae

    def xgboost(self, x_train, x_test, y_train, y_test):
        model=XGBRegressor(objective="reg:squarederror",n_estimators=2000,seed=123,max_depth=5,learning_rate=0.1,random_state=42)
        model.fit(x_train,y_train)
        prediction=model.predict(x_test)
        r2=r2_score(y_test,prediction)
        rmse=root_mean_squared_error(y_test,prediction)
        mse=mean_squared_error(y_test,prediction)
        mae=mean_absolute_error(y_test,prediction)
        
        print(f"XGBoost - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
        mlflow.set_experiment("clickstreamregression")
        with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "xgboost")
        return model, r2,mae

    def evaluate_models(self, x_train, x_test, y_train, y_test):
        models = {
            "Linear Regression": self.linear_regression,
            "Lasso Regression": self.lasso_regression,
            "Ridge Regression": self.ridge_regression,
            "Gradient Boosting": self.gradient_boosting_regressor,
            "XGBoost": self.xgboost
        }

        best_model = None
        best_r2 = float('-inf')
        best_mae=float('inf')
        best_model_name = ""

        for name, model_func in models.items():
            trained_model, r2 ,mae= model_func(x_train, x_test, y_train, y_test)
            if r2 > best_r2 and mae<best_mae:
                best_r2 = r2
                best_mae=mae
                best_model = trained_model
                best_model_name = name  
        with open(f"{best_model_name}_clickstream.pkl", "wb") as f:
            pkl.dump(best_model, f)
        print(f"Best Model: {best_model_name} with R2 Score: {best_r2:.4f}")


