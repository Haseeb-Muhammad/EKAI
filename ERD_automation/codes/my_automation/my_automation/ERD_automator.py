import os
import pandas as pd
from attribute import Attribute
from ind import IND
import openai
import logging
from Database import Database
import json

class ERD_automator:
    def __init__(self, database: Database):
        self.database = database

        self.attributes = {}
        self.primary_keys = set() #contains primary keys in the form tablename.columnname
        self.INDs = []
        self.candidateKeys = []

        # Loading attributes and INDs
        self.load_attributes()
        self.load_INDs()

    
    def load_attributes(self) -> None:
        csv_files = [f for f in os.listdir(self.database.database_dir)]
        logging.info(f"Found {len(csv_files)} \n CSV files: {csv_files}")
        logging.info("-"*50)

        for filename in csv_files:
            file_path = os.path.join(self.database.database_dir, filename)
            table_name = os.path.splitext(filename)[0]

            df = pd.read_csv(file_path)
            logging.info(f"Processing {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

            for i, column in enumerate(df.columns):
                non_null_values = df[column].astype(str).dropna().tolist()
                if non_null_values:
                    attr = Attribute(table_name, column, non_null_values)
                    attr.position = 1/(i+1)
                    self.attributes[f"{table_name}.{column}"] = attr
                    logging.info(f"Added attribute: {attr.table_name}.{attr.attribute_name} Total Values: {len(attr.values)}")            
        logging.info("-"*50) 


    def load_INDs(self):
        logging.info("Reading INDs from SPIDER")
        logging.info("-"*50)
        with open(self.database.database_dir, "r") as f:
            for line in f:
                vars = line.strip().split("=")
                self.inds.append(IND(dependent=self.ttributes[vars[1]], reference=self.attributes[vars[0]]))
        logging.info(f"Number of INDs from spider: {len(self.inds)}")


    def evaluate(self):
        with open(self.database.gt_path) as file_data:
            json_data = json.load(file_data)
            relations = json_data["relations"]

            results = {
                            "TP" : [],
                            "FP" : [],
                            "FN" : [],
                        }
                
            #Convert predictions to set for O(1) look up
            pred_set = set()
            pred_dict = {} 
            for ind in self.inds:
                key = f"{ind.reference.fullName}={ind.dependent.fullName}"
                pred_set.add(key)
                pred_dict[key] = ind

            for gt_key in relations:
                relation_name = f"{gt_key["primary_key_table_column"].lower()}={gt_key["foreign_key_table_column"].lower()}"
                if relation_name in pred_set:
                    results["TP"].append(relation_name)
                    pred_set.remove(relation_name)
                else:
                    results["FN"].append(relation_name)
                    print(f"Added {gt_key} to FN")

            results["FP"] = list(pred_set)


        logging.info(f"{"-"*50} Results {"-"*50}")
        logging.info("TP")
        for tp in results["TP"]:
            logging.info(tp)
        logging.info("FP")
        for fp in results["FP"]:
            logging.info(fp)
        logging.info("FN")
        for fn in results["FN"]:
            logging.info(fn)
        logging.info(f"{"-"*50} Evaluations {"-"*50}")
        logging.info(f"True Positive: {len(results["TP"])}")
        logging.info(f"False Positive: {len(results["FP"])}")
        logging.info(f"False Negative: {len(results["FN"])}")

        return results
