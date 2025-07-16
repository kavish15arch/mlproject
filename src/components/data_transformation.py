import sys
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from exception import CustomException
from utils import save_object
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
import os



@dataclass # @dataclass is a decorator in Python (from the dataclasses module) that makes it easy to create classes which are mainly used to store data.Iska use krne se hame variable ko initialize nhi karna padta hai directly kaam ho jata hai
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts' , 'preproccesor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            num_features=[ 'reading_score', 'writing_score']
            cat_features=['gender',
                          'race_ethnicity',
                         'parental_level_of_education',
                          'lunch',
                         'test_preparation_course']
        
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),# imputer handles the missing values
                    ('standard scalar',StandardScaler())
                ]
            )


            cat_features_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder',OneHotEncoder())
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('numerical pipeline', num_pipeline,num_features),
                    ('categorical pipeline', cat_features_pipeline,cat_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise e
        
    def initiate_data_transformation(self, train_df , test_df):

        try:
            train_path=train_df
            test_path=test_df

            

            preprocessing_obj=self.get_data_transformer_obj()

            target_score='math_score'

            num_features=[ 'reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_score],axis=1)
            target_feature_train_df=train_df[target_score]

            input_feature_test_df=test_df.drop(columns=[target_score],axis=1)
            target_feature_test_df=test_df[target_score]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr=np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)

            ]#c_ is used to combine the columns

            test_arr=np.c_[input_feature_test_arr , np.array(target_feature_test_df)]


            save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
    )


           

            return (train_arr,test_arr,preprocessing_obj
)

        except Exception as e:
            raise e


           