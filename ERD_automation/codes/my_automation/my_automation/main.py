import os
import pandas as pd
from attribute import Attribute
from ind import IND
import logging


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
        if pk_table[ind.reference.table_name][0] == ind.reference.fullName:
            is_pk=True

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

def auto_incremental_pk_pruning(inds):
    """
    Prunes inclusion dependencies (INDs) that represent auto-incremental primary keys.

    This function checks each IND in the input list and removes those where the dependent values
    form a consecutive subsequence within the reference values, which is indicative of an auto-incremental
    primary key relationship.

    Parameters
    ----------
    inds : list
        A list of IND objects, each containing 'dependent' and 'reference' attributes with 'values' lists.

    Returns
    -------
    pruned_inds : list
        A list of IND objects after removing those that represent auto-incremental primary keys.

    Notes
    -----
    The function logs the number of INDs remaining after pruning.
    """
    pruned_inds = []
    for ind in inds:
        i =0
        length_dependent = len(ind.dependent.values)
        length_reference = len(ind.reference.values)
        subset = False
        while i <= (length_reference-length_dependent + 1):
            if not (ind.reference.values[i:i+length_dependent] == ind.dependent.values):
                subset = True
            i+=1
        if not subset:
            pruned_inds.append(ind)
    logging.info(f"Number of INDs after pruning auto incremental primary keys: {len(pruned_inds)}")
    return pruned_inds

def check_multi_dependence(inds):
    """
    Prunes a list of INDs by removing those that reference the same dependent attribute multiple times.

    Parameters
    ----------
    inds : list
        A list of IND objects, each with a 'dependent' attribute.

    Returns
    -------
    list
        A pruned list of INDs where no dependent attribute is referenced by more than one IND.

    Notes
    -----
    This function ensures that the same dependent attribute is not referenced by multiple INDs,
    which is useful for maintaining integrity when handling auto-incremental primary keys.
    """
    pruned_inds = []
    for ind in inds:
        multi_referenced = False
        for ind1 in inds:
            if ind!= ind1 and ind.dependent == ind1.dependent:
                multi_referenced = True
        if not multi_referenced:
            pruned_inds.append(ind)
    logging.info(f"Number of INDs after pruning multi-dependencies keys: {len(pruned_inds)}")
    return pruned_inds

def check_dependent_referencing(inds):
    """
    Removes elements from the input list whose dependent attribute references another element's reference attribute.
    Parameters
    ----------
    inds : list
        A list of objects, each expected to have 'dependent' and 'reference' attributes.
    Returns
    -------
    new_inds : list
        A list containing only those elements from `inds` whose dependent attribute does not reference any other element's reference attribute.
    Notes
    -----
    The function does not modify the input list in place; instead, it returns a new list.
    Logging is performed to indicate the number of elements after pruning.
    """
    new_inds = [] #Can't change the list which is being looped over
    for ind in inds:
        dependent_referencing=False
        for ind1 in inds:
            if (ind!=ind1) and (ind.dependent == ind1.reference):
                dependent_referencing=True
        if not dependent_referencing:
            new_inds.append(ind)
    logging.info(f"Number of INDs after pruning dependent variable referencing others: {len(new_inds)}")
    
    return new_inds

def logINDs(inds):
    """
    Logs the relationships between reference and dependent objects in a list of INDs.

    Parameters
    ----------
    inds : list
        A list of IND objects, each containing 'reference' and 'dependent' attributes with 'fullName' properties.

    """
    logging.info("-"*50)
    for ind in inds:
        logging.info(f"{ind.reference.fullName}->{ind.dependent.fullName}")

def main():
    CSV_DIR = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind-db"
    SPIDER_IND_RESULT = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/northwind.txt"


    db_name=os.path.basename(CSV_DIR)
    logging.basicConfig(
                    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),f"{db_name}.log"),
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=10
    )


    attributes = load_csv_files(CSV_DIR)
    pk_table = extractPrimaryKeys(attributes=attributes)   
    inds = read_IND(SPIDER_IND_RESULT, attributes=attributes)
    prefiltered_inds = prefiltering(inds=inds, pk_table=pk_table)
    pruned_inds = auto_incremental_pk_pruning(prefiltered_inds)
    dependent_referencing_filtered_inds = check_dependent_referencing(pruned_inds)
    logINDs(dependent_referencing_filtered_inds)

if __name__ == "__main__":
    main()
            
