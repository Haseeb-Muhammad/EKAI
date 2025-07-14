import os
import csv
import heapq
from collections import defaultdict
import pandas as pd

def load_csv_files(directory_path):
    """
    Load all CSV files from the given directory and extract column data.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        
    Returns:
        dict: Dictionary mapping 'table.column' to sorted unique values
    """
    column_dict = {}
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    for filename in csv_files:
        filepath = os.path.join(directory_path, filename)
        table_name = os.path.splitext(filename)[0]  # Remove .csv extension
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Process each column
            for column in df.columns:
                column_key = f"{table_name}.{column}"
                
                # Get unique values, remove NaN, and sort
                unique_values = df[column].dropna().unique()
                sorted_values = sorted(unique_values)
                
                column_dict[column_key] = sorted_values
                print(f"Loaded {len(sorted_values)} unique values from {column_key}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return column_dict

def initialize_inclusion_dict(column_dict):
    """
    Initialize inclusion dictionary where each column can potentially 
    be included in all other columns.
    
    Args:
        column_dict (dict): Dictionary of column names to their values
        
    Returns:
        dict: Initial inclusion dictionary
    """
    columns = list(column_dict.keys())
    inclusion_dict = {}
    
    for col in columns:
        # Initially, each column could be included in all other columns
        inclusion_dict[col] = [other_col for other_col in columns if other_col != col]
    
    return inclusion_dict

def spider_algorithm(column_dict):
    """
    Spider algorithm implementation using min-heap to find inclusion dependencies.
    
    Args:
        column_dict (dict): Dictionary mapping column names to sorted unique values
        
    Returns:
        dict: Final inclusion dictionary showing dependencies
    """
    # Initialize inclusion dictionary
    inclusion_dict = initialize_inclusion_dict(column_dict)
    
    # Initialize min heap with all values from all columns
    min_heap = []
    
    print("Building heap...")
    for column in column_dict:
        vals = column_dict[column]
        for val in vals:
            tup = (str(val), column)
            heapq.heappush(min_heap, tup)
    
    print(f"Heap initialized with {len(min_heap)} elements")
    
    # Process heap
    iteration = 0
    while min_heap:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"Processing iteration {iteration}, heap size: {len(min_heap)}")
        
        # Get the smallest value in the heap
        att = []
        current_smallest, var = heapq.heappop(min_heap)
        att.append(var)
        
        # Pop all elements where values are equal to current smallest
        while min_heap and min_heap[0][0] == current_smallest:
            next_var = heapq.heappop(min_heap)[-1]
            att.append(next_var)
        
        # Update inclusion_dict
        # For each attribute in att, it can only be included in other attributes in att
        for a in att:
            if a in inclusion_dict:
                inclusion_dict[a] = list(set(inclusion_dict[a]).intersection(att))
    
    print(f"Algorithm completed after {iteration} iterations")
    return inclusion_dict

def filter_inclusion_dependencies(inclusion_dict):
    """
    Filter inclusion dependencies to remove self-references and empty lists.
    
    Args:
        inclusion_dict (dict): Raw inclusion dictionary
        
    Returns:
        dict: Filtered inclusion dictionary
    """
    filtered_dict = {}
    
    for dependent, references in inclusion_dict.items():
        # Remove self-references and filter non-empty lists
        filtered_references = [ref for ref in references if ref != dependent]
        if filtered_references:
            filtered_dict[dependent] = filtered_references
    
    return filtered_dict

def write_output(inclusion_dict, output_file):
    """
    Write inclusion dependencies to output file in the specified format.
    
    Args:
        inclusion_dict (dict): Dictionary of inclusion dependencies
        output_file (str): Path to output file
    """
    with open(output_file, 'w') as f:        
        # Sort for consistent output
        for dependent in sorted(inclusion_dict.keys()):
            references = inclusion_dict[dependent]
            for reference in sorted(references):
                f.write(f"{reference}={dependent}\n")
    
    print(f"Output written to {output_file}")

def find_inclusion_dependencies(directory_path, output_file="inclusion_dependencies.txt"):
    """
    Main function to find inclusion dependencies from CSV files.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        output_file (str): Path to output file (default: "inclusion_dependencies.txt")
    """
    try:
        print(f"Starting Spider algorithm for directory: {directory_path}")
        
        # Load CSV files and extract column data
        column_dict = load_csv_files(directory_path)
        
        if not column_dict:
            print("No data loaded. Exiting.")
            return
        
        print(f"Loaded {len(column_dict)} columns from CSV files")
        
        # Run Spider algorithm
        inclusion_dict = spider_algorithm(column_dict)
        
        # Filter results
        filtered_dict = filter_inclusion_dependencies(inclusion_dict)
        
        # Write output
        write_output(filtered_dict, output_file)
        
        print(f"Found {sum(len(refs) for refs in filtered_dict.values())} inclusion dependencies")
        
        # Print summary
        print("\nSummary of inclusion dependencies:")
        for dependent in sorted(filtered_dict.keys()):
            references = filtered_dict[dependent]
            for reference in sorted(references):
                print(f"  {reference} = {dependent}")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    # Example usage - replace with your directory path
    directory_path = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/sakila-db"
    output_file = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/sakila.txt"
find_inclusion_dependencies(directory_path, output_file)

