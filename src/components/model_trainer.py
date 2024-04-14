import os

import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:

    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGB Classifier": XGBClassifier(),
                #"GridSearch": GridSearchCV(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "SVC": SVC(),
                "Kneighbourclassifier":KNeighborsClassifier(),
                "Adaboost":AdaBoostClassifier(),
                "Random forest":RandomForestClassifier(),
                "Bernoullinb":BernoulliNB()
            }
            
            '''params={
                "Logistic Regression" : {
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 200, 300]
                }

                "Decision Tree" : {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

                "Gradient Boosting" :{
                    'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.5, 0.8, 1.0]
                } 

                #"XGB Classifier" :{
                    'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [50, 100, 200],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.5, 0.8, 1.0],
                    'colsample_bytree': [0.5, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }

                "CatBoosting Regressor" = {
                    'learning_rate': [0.01, 0.1, 0.5],
                    'iterations': [50, 100, 200],
                    'depth': [3, 5, 7],
                    'l2_leaf_reg': [1, 3, 5],
                    'bagging_temperature': [0.5, 0.8, 1.0],
                    'random_strength': [0.1, 0.5, 1.0],
                    'border_count': [32, 64, 128]
                }

                "SVC" :{
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4],
                    'coef0': [0.0, 0.1, 0.5]
                }

                "Kneighbourclassifier": {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 30, 50],
                    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
                }

                "Adaboost" :{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }

                "Random forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }

                "Bernoullinb" :{
                    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
                    'binarize': [None, 0.0, 0.5, 1.0]
                }'''


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)