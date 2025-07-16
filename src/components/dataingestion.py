import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transformation import DataTransformationConfig, DataTransformation
from model_trainer import ModelTrainer, ModelTrainerConfig


class DataIngestionConfig:
    raw_data_path = "artifacts/raw_data.csv"
    train_data_path = "artifacts/train_data.csv"
    test_data_path = "artifacts/test_data.csv"

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv("data.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
           pass


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    train_df=pd.read_csv(train_data)
    test_df=pd.read_csv(test_data)

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_df,test_df)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))





    

