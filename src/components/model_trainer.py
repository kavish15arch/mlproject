import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from exception import CustomException
from logger import logging
from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting train and test data')
            X_train= train_array[:,:-1]
            y_train=train_array[:,-1]
            X_test=test_array[:,:-1]
            y_test=test_array[:,-1]
            
            models={
                     'Linear':LinearRegression(),
                         'Lasso':Lasso(),
                        'Ridge':Ridge(),
                        'ranadom forest':RandomForestRegressor(),
                            'adaboost':AdaBoostRegressor(),
                              'k neighbour':KNeighborsRegressor(),
                              'Adaboost':AdaBoostRegressor(),
                              'gradient boosting regressor':GradientBoostingRegressor()
                                                                }
            
            params={'Linear':{},
                    'k neighbour':{'weights':['uniform', 'distance'],
                                           'n_neighbors':[5,7,9,10,11],
                                           'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                                           },
                    'Adaboost':{'n_estimators':[32,16,64,128,256],
                                'learning_rate':[0.1,0.01,0.5,1.0,0.001],
                                'loss':['linear', 'square', 'exponential']},
                    'gradient boosting regressor':{'loss':['log_loss','exponential'],
                                                   'learning_rate':[0.1,0.01,0.5,1.0,0.001],
                                                   'n_estimators':[32,16,64,128,256]
                                                   },
                    'random forest':{'n_estimators':[32,16,64,128,256],
                                     'criterion':['gini', 'entropy', 'log_loss'],
                                     'max_features':['sqrt', 'log2', None]},
                    'Ridge':{},
                    'Lasso':{}             
                                      }
            

            model_score:dict=evaluate_models(X_train=X_train ,X_test=X_test,y_train=y_train,y_test=y_test,models=models,param=params)


            best_model_score=max(model_score.values())
            best_model_name = list(model_score.keys())[list(model_score.values()).index(best_model_score)]

            best_model=models[best_model_name]

            

            if best_model_score<0.6:
                raise CustomException('no best model found')
            

            save_object(file_path=self.model_trainer_config.trained_model_file_path , obj=best_model)

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise e
            


       
            
       
    

 
            

