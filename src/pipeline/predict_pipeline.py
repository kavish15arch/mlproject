import sys
import pandas as pd
from src.components.utils import load_object

class PredictPipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            data_pred=model.pred(data_scaled)
            return data_pred
        except Exception as e:
            raise e


class CustomData:
    def __init(self,gender:str,race_ethinicity:int,parental_level_of_education,lunch:str,test_prepration_course:str,reading_score:int,writing_score:int):
        self.gender=gender
        self.race_ethincity=race_ethinicity
        self.parental_level_of_edeucation=parental_level_of_education
        self.lunch=lunch
        self.test_prepration_course=test_prepration_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def get_data_as_data_frame(self):
        try:
            cutom_data_input_dict={'gender':[self.gender],'race_ethinicity':[self.race_ethincity],'parental_level_of_education':[self.parental_level_of_edeucation],'lunch':[self.lunch],'test_prepration_course':[self.test_prepration_course],'reading_score':[self.reading_score],'writing_score':[self.writing_score]}

            return pd.DataFrame(cutom_data_input_dict)
        
        except Exception as e:
            raise e
