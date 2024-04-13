import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 age: int, sex: int, cp: int,trtbs: int,chol: int,
                 fbs:int,restecg: int,thalachh: int,exng: int,
                 oldpeak: int,slp: int,caa: int,
                 thall: int):
        
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trtbs = trtbs
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalachh = thalachh
        self.exng= exng
        self.oldpeak = oldpeak
        self.slp = slp
        self.caa = caa
        self.thall = thall


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age":[self.age],
                "Sex":[self.sex],
                "cp":[self.cp],
                "trtbs":[self.trtbs],
                "chol":[self.chol],
                "fbs":[self.fbs],
                "restecg":[self.restecg],
                "thalachh":[self.thalachh],
                "exng":[self.exng],
                "oldpeak":[self.oldpeak],
                "slp":[self.slp],
                "caa":[self.caa],
                "thall":[self.thall],

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)







