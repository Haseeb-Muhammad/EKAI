import os
import pandas as pd
from attribute import Attribute
from ind import IND
import logging

logging.basicConfig(
                    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),"app.log"),
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=10
)

def load_csv_files(directory_path):
    """
    Load CSV files from a directory and create Attribute objects.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing CSV files
        
    Returns
    -------
    dict
        Dictionary mapping "table.column" strings to Attribute objects
        
    Notes
    -----
    Prints progress information during loading
    """
    attributes = {}

    csv_files = [f for f in os.listdir(directory_path)]

    logging.info(f"Found {len(csv_files)} \n CSV files: {csv_files}")

    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        table_name = os.path.splitext(filename)[0]

        df = pd.read_csv(file_path)
        logging.info(f"Processing {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

        for i, column in enumerate(df.columns):
            non_null_values = df[column].astype(str).dropna().tolist()
            if non_null_values:
                attr = Attribute(table_name, column, non_null_values)
                attr.position = 1/(i+1)
                attributes[f"{table_name}.{column}"] = attr
                logging.info(f"Added attribute: {attr.table_name}.{attr.attribute_name} Total Values: {len(attr.values)}")
                
    logging.info("------------------------------------------------------")
    return attributes

def extractPrimaryKeys(attributes):
    """
    Extract primary keys from attributes based on their pkScore.

    Parameters
    ----------
    attributes : dict
        Dictionary mapping "table.column" strings to Attribute objects

    Returns
    -------
    dict
        Dictionary mapping table names to tuples of (column name, pkScore)
    """
    pk_table = {}  # {table name: (tableName.AttributeName, score)}

    for key, value in attributes.items():
        table_name = key.split(".")[0]
        current_pk = pk_table.get(table_name)
        if not current_pk or current_pk[1] < value.pkScore:
            pk_table[table_name] = (value.fullName, value.pkScore)
    logging.info("--------------------------Tables and their Primary Keys----------------------------")
    for key, value in pk_table.items():
        logging.info(f"{key} : {value[0]}")
    logging.info("------------------------------------------------------")
    return pk_table

def read_IND(file_path, attributes):
    """
    Read inclusion dependencies from a file.
    
    Parameters
    ----------
    file_path : str
        Path to file containing inclusion dependencies
        
    Returns
    -------
    list of IND
        List of inclusion dependency objects
        
    Notes
    -----
    File format should be one dependency per line as: dependent=reference
    """
    inds = []
    with open(file_path, "r") as f:
        for line in f:
            vars = line.strip().split("=")
            inds.append(IND(attributes[vars[0]], attributes[vars[1]]))
    logging.info(f"Number of INDs from spider: {len(inds)}")
    return inds

def prefiltering(inds, pk_table):
    """
    Filter inclusion dependencies based on primary key and null value criteria.
    
    Parameters
    ----------
    inds : list of IND
        List of inclusion dependency objects to filter
        
    Returns
    -------
    list of IND
        Filtered list of inclusion dependencies that meet criteria:
        - Reference attribute is a primary key
        - Neither dependent nor reference is all null values
        
    Notes
    -----
    Uses global pk_table dictionary for primary key lookup
    """
    pruned_inds = []
    for ind in inds:
        #Checking if reference variable is a primary key
        is_pk = False
        for table_name, pk in pk_table.items():
            if pk[0].split(".")[1] == ind.reference.attribute_name:
                is_pk=True
                break

        #Checking if either all of the dependent or reference attribute is null
        dependent_all_null = True
        reference_all_null = True
        for value in ind.reference.values:
            if value != "nan":
                reference_all_null = False
        for value in ind.dependent.values:
            if value !="nan":
                dependent_all_null = False
        
        if is_pk and (not reference_all_null) and (not dependent_all_null):
            pruned_inds.append(ind)
    logging.info(f"Number of INDs after prefiltering: {len(pruned_inds)}")

    return pruned_inds

def main():
    CSV_DIR = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind-db"
    SPIDER_IND_RESULT = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/northwind.txt"
    
    attributes = load_csv_files(CSV_DIR)
    pk_table = extractPrimaryKeys(attributes=attributes)   
    inds = read_IND(SPIDER_IND_RESULT, attributes=attributes)
    prefiltered_inds = prefiltering(inds=inds, pk_table=pk_table)

if __name__ == "__main__":
    main()
            
