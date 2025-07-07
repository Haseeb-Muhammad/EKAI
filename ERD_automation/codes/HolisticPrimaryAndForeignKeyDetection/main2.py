import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Set, Optional
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HoPFCSVDetector:
    """
    Holistic Primary Key and Foreign Key Detection for CSV Database Files
    
    This class implements the HoPF algorithm for automatically detecting
    primary keys and foreign keys from CSV files in a directory.
    """
    
    def __init__(self, csv_directory: str, encoding: str = 'utf-8', 
                 max_combination_size: int = 3, min_rows_threshold: int = 2):
        """
        Initialize the HoPF detector with CSV files from a directory.
        
        Args:
            csv_directory: Path to directory containing CSV files
            encoding: CSV file encoding (default: utf-8)
            max_combination_size: Maximum number of columns to consider for combinations
            min_rows_threshold: Minimum number of rows required to consider a table
        """
        self.csv_directory = Path(csv_directory)
        self.encoding = encoding
        self.max_combination_size = max_combination_size
        self.min_rows_threshold = min_rows_threshold
        
        self.tables = {}  # Dictionary of table_name -> DataFrame
        self.table_info = {}  # Metadata about each table
        self.uccs = {}  # Unique Column Combinations
        self.inds = []  # Inclusion Dependencies
        self.primary_keys = {}
        self.foreign_keys = {}
        
        self._load_csv_files()
        
    def _load_csv_files(self):
        """Load all CSV files from the specified directory."""
        if not self.csv_directory.exists():
            raise FileNotFoundError(f"Directory {self.csv_directory} does not exist")
        
        csv_files = list(self.csv_directory.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_directory}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Use filename without extension as table name
                table_name = csv_file.stem
                
                # Try different encodings if utf-8 fails
                encodings_to_try = [self.encoding, 'latin-1', 'cp1252', 'iso-8859-1']
                
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                        logger.info(f"Successfully loaded {table_name} with encoding {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    logger.warning(f"Failed to load {csv_file} with any encoding")
                    continue
                
                # Skip tables with insufficient data
                if len(df) < self.min_rows_threshold:
                    logger.warning(f"Skipping {table_name}: only {len(df)} rows")
                    continue
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Store table and metadata
                self.tables[table_name] = df
                self.table_info[table_name] = {
                    'file_path': str(csv_file),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns)
                }
                
                logger.info(f"Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")
                continue
        
        if not self.tables:
            raise ValueError("No valid CSV files could be loaded")
        
        logger.info(f"Successfully loaded {len(self.tables)} tables")
    
    def detect_keys(self) -> Dict[str, Dict]:
        """
        Main method to detect primary keys and foreign keys.
        
        Returns:
            Dictionary containing detected primary keys and foreign keys
        """
        logger.info("Starting key detection...")
        
        # Step 1: Discover UCCs for each table
        logger.info("Discovering Unique Column Combinations...")
        self._discover_uccs()
        
        # Step 2: Discover INDs between tables
        logger.info("Discovering Inclusion Dependencies...")
        self._discover_inds()
        
        # Step 3: Score and select primary keys
        logger.info("Selecting Primary Keys...")
        self._select_primary_keys()
        
        # Step 4: Score and select foreign keys
        logger.info("Selecting Foreign Keys...")
        self._select_foreign_keys()
        
        logger.info("Key detection completed!")
        
        return {
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys,
            'table_info': self.table_info
        }
    
    def _discover_uccs(self):
        """Discover Unique Column Combinations (UCCs) for each table."""
        for table_name, table in self.tables.items():
            logger.info(f"Finding UCCs for {table_name}...")
            uccs = []
            columns = list(table.columns)
            
            # Check combinations up to max_combination_size
            for size in range(1, min(len(columns) + 1, self.max_combination_size + 1)):
                for combo in combinations(columns, size):
                    if self._is_unique_combination(table, list(combo)):
                        # Only add if not a superset of existing minimal UCC
                        if not self._is_superset_of_existing_ucc(list(combo), uccs):
                            uccs.append(list(combo))
                            logger.debug(f"Found UCC in {table_name}: {combo}")
            
            self.uccs[table_name] = uccs
            logger.info(f"Found {len(uccs)} UCCs for {table_name}")
    
    def _is_unique_combination(self, table: pd.DataFrame, columns: List[str]) -> bool:
        """Check if a combination of columns uniquely identifies rows."""
        if not all(col in table.columns for col in columns):
            return False
        
        # Remove rows with null values in any of the key columns
        subset = table[columns].dropna()
        if len(subset) == 0:
            return False
        
        # Check if combination is unique
        return len(subset) == len(subset.drop_duplicates())
    
    def _is_superset_of_existing_ucc(self, combo: List[str], existing_uccs: List[List[str]]) -> bool:
        """Check if a combination is a superset of any existing UCC."""
        combo_set = set(combo)
        for ucc in existing_uccs:
            if set(ucc).issubset(combo_set) and set(ucc) != combo_set:
                return True
        return False
    
    def _discover_inds(self):
        """Discover Inclusion Dependencies (INDs) between tables."""
        table_names = list(self.tables.keys())
        
        for i, table1_name in enumerate(table_names):
            table1 = self.tables[table1_name]
            for j, table2_name in enumerate(table_names):
                if i == j:
                    continue
                
                table2 = self.tables[table2_name]
                logger.debug(f"Checking INDs between {table1_name} and {table2_name}...")
                
                # Check single column INDs
                for col1 in table1.columns:
                    for col2 in table2.columns:
                        if self._is_inclusion_dependency(table1, [col1], table2, [col2]):
                            score = self._calculate_ind_score(table1, [col1], table2, [col2])
                            self.inds.append({
                                'dependent': (table1_name, [col1]),
                                'referenced': (table2_name, [col2]),
                                'score': score
                            })
                            logger.debug(f"Found IND: {table1_name}.{col1} -> {table2_name}.{col2} (score: {score:.3f})")
                
                # Check multi-column INDs (up to max_combination_size)
                for size in range(2, min(self.max_combination_size + 1, 
                                       min(len(table1.columns), len(table2.columns)) + 1)):
                    for col1_combo in combinations(table1.columns, size):
                        for col2_combo in combinations(table2.columns, size):
                            if self._is_inclusion_dependency(table1, list(col1_combo), 
                                                          table2, list(col2_combo)):
                                score = self._calculate_ind_score(table1, list(col1_combo), 
                                                                table2, list(col2_combo))
                                self.inds.append({
                                    'dependent': (table1_name, list(col1_combo)),
                                    'referenced': (table2_name, list(col2_combo)),
                                    'score': score
                                })
                                logger.debug(f"Found multi-column IND: {table1_name}.{col1_combo} -> {table2_name}.{col2_combo} (score: {score:.3f})")
        
        logger.info(f"Found {len(self.inds)} inclusion dependencies")
    
    def _is_inclusion_dependency(self, table1: pd.DataFrame, cols1: List[str], 
                               table2: pd.DataFrame, cols2: List[str]) -> bool:
        """Check if columns in table1 are included in columns of table2."""
        if len(cols1) != len(cols2): # why is this importnat?
            return False
        
        # Get unique values from both tables (excluding nulls)
        values1 = set()
        values2 = set()
        
        # Handle single vs multiple columns
        if len(cols1) == 1:
            values1 = set(table1[cols1[0]].dropna().unique())
            values2 = set(table2[cols2[0]].dropna().unique())
        else:
            for _, row in table1[cols1].dropna().iterrows():
                values1.add(tuple(row[cols1]))
            for _, row in table2[cols2].dropna().iterrows():
                values2.add(tuple(row[cols2]))
        
        if len(values1) == 0 or len(values2) == 0:
            return False
        
        # Check if all values in table1 are included in table2
        inclusion_ratio = len(values1.intersection(values2)) / len(values1)
        return inclusion_ratio >= 0.95  # Allow for some data inconsistencies
    
    def _calculate_ind_score(self, table1: pd.DataFrame, cols1: List[str], 
                           table2: pd.DataFrame, cols2: List[str]) -> float:
        """Calculate inclusion dependency score."""
        # Get unique values
        if len(cols1) == 1:
            values1 = set(table1[cols1[0]].dropna().unique())
            values2 = set(table2[cols2[0]].dropna().unique())
        else:
            values1 = set()
            values2 = set()
            for _, row in table1[cols1].dropna().iterrows():
                values1.add(tuple(row[cols1]))
            for _, row in table2[cols2].dropna().iterrows():
                values2.add(tuple(row[cols2]))
        
        if len(values1) == 0:
            return 0.0
        
        # Coverage: ratio of matching values
        coverage = len(values1.intersection(values2)) / len(values1)
        
        # Cardinality ratio (prefer many-to-one relationships)
        cardinality_score = min(len(values2) / max(len(values1), 1), 1.0)
        
        # Column name similarity
        name_similarity = self._calculate_name_similarity(cols1, cols2)
        
        # Data type compatibility
        dtype_compatibility = self._calculate_dtype_compatibility(table1, cols1, table2, cols2)
        
        return (coverage * 0.5 + cardinality_score * 0.2 + 
                name_similarity * 0.15 + dtype_compatibility * 0.15)
    
    def _calculate_name_similarity(self, cols1: List[str], cols2: List[str]) -> float:
        """Calculate name similarity between column sets."""
        if len(cols1) != len(cols2):
            return 0.0
        
        similarity = 0.0
        for c1, c2 in zip(cols1, cols2):
            c1_lower = c1.lower().strip()
            c2_lower = c2.lower().strip()
            
            if c1_lower == c2_lower:
                similarity += 1.0
            elif c1_lower in c2_lower or c2_lower in c1_lower:
                similarity += 0.7
            elif any(word in c2_lower for word in c1_lower.split('_')):
                similarity += 0.5
        
        return similarity / len(cols1)
    
    def _calculate_dtype_compatibility(self, table1: pd.DataFrame, cols1: List[str], 
                                     table2: pd.DataFrame, cols2: List[str]) -> float:
        """Calculate data type compatibility score."""
        if len(cols1) != len(cols2):
            return 0.0
        
        compatibility = 0.0
        for c1, c2 in zip(cols1, cols2):
            dtype1 = str(table1[c1].dtype)
            dtype2 = str(table2[c2].dtype)
            
            if dtype1 == dtype2:
                compatibility += 1.0
            elif ('int' in dtype1 and 'int' in dtype2) or ('float' in dtype1 and 'float' in dtype2):
                compatibility += 0.8
            elif ('object' in dtype1 and 'object' in dtype2):
                compatibility += 0.6
        
        return compatibility / len(cols1)
    
    def _select_primary_keys(self):
        """Select primary keys from UCCs using scoring function."""
        for table_name, uccs in self.uccs.items():
            if not uccs:
                logger.warning(f"No UCCs found for {table_name}")
                continue
            
            # Score each UCC
            scored_uccs = []
            table = self.tables[table_name]
            
            for ucc in uccs:
                score = self._calculate_pk_score(table, ucc, table_name)
                scored_uccs.append((ucc, score))
            
            # Sort by score (descending) and select the best one
            scored_uccs.sort(key=lambda x: x[1], reverse=True)
            
            if scored_uccs:
                best_pk = scored_uccs[0][0]
                self.primary_keys[table_name] = best_pk
                logger.info(f"Selected PK for {table_name}: {best_pk} (score: {scored_uccs[0][1]:.3f})")
    
    def _calculate_pk_score(self, table: pd.DataFrame, ucc: List[str], table_name: str) -> float:
        """Calculate primary key score for a UCC."""
        # Prefer shorter keys
        length_score = 1.0 / len(ucc)
        
        # Prefer keys with 'id' in name
        name_score = 0.0
        for col in ucc:
            col_lower = col.lower()
            if col_lower.endswith('id') or col_lower.startswith('id'):
                name_score += 0.5
            elif 'id' in col_lower:
                name_score += 0.3
        name_score = min(name_score, 1.0)
        
        # Prefer keys with no nulls
        null_score = 1.0
        for col in ucc:
            if col in table.columns:
                null_ratio = table[col].isnull().sum() / len(table)
                null_score *= (1 - null_ratio)
        
        # Prefer numeric keys
        numeric_score = 0.0
        for col in ucc:
            if col in table.columns:
                if pd.api.types.is_numeric_dtype(table[col]):
                    numeric_score += 0.3
        numeric_score = min(numeric_score, 1.0)
        
        # Bonus for being referenced by other tables
        reference_score = 0.0
        for ind in self.inds:
            if (ind['referenced'][0] == table_name and 
                set(ind['referenced'][1]) == set(ucc)):
                reference_score += 0.1
        reference_score = min(reference_score, 1.0)
        
        return (length_score * 0.3 + name_score * 0.25 + null_score * 0.25 + 
                numeric_score * 0.1 + reference_score * 0.1)
    
    def _select_foreign_keys(self):
        """Select foreign keys from INDs using scoring function."""
        # Sort INDs by score
        sorted_inds = sorted(self.inds, key=lambda x: x['score'], reverse=True)
        
        selected_fks = []
        
        for ind in sorted_inds:
            dependent_table, dependent_cols = ind['dependent']
            referenced_table, referenced_cols = ind['referenced']
            
            # Check if referenced columns form a primary key or UCC
            is_pk_reference = (referenced_table in self.primary_keys and 
                             set(referenced_cols) == set(self.primary_keys[referenced_table]))
            
            is_ucc_reference = (referenced_table in self.uccs and 
                              any(set(referenced_cols) == set(ucc) for ucc in self.uccs[referenced_table]))
            
            if (is_pk_reference or is_ucc_reference) and self._is_valid_foreign_key(ind):
                selected_fks.append(ind)
                logger.info(f"Selected FK: {dependent_table}.{dependent_cols} -> {referenced_table}.{referenced_cols} (score: {ind['score']:.3f})")
        
        # Group by dependent table
        fk_by_table = defaultdict(list)
        for fk in selected_fks:
            fk_by_table[fk['dependent'][0]].append(fk)
        
        self.foreign_keys = dict(fk_by_table)
    
    def _is_valid_foreign_key(self, ind: Dict) -> bool:
        """Additional validation for foreign key candidates."""
        dependent_table, dependent_cols = ind['dependent']
        referenced_table, referenced_cols = ind['referenced']
        
        # Don't allow self-referencing (can be extended later)
        if dependent_table == referenced_table:
            return False
        
        # Check if the score is above threshold
        if ind['score'] < 0.6:
            return False
        
        # Check for reasonable cardinality
        table1 = self.tables[dependent_table]
        table2 = self.tables[referenced_table]
        
        if len(dependent_cols) == 1:
            unique_vals1 = table1[dependent_cols[0]].nunique()
            unique_vals2 = table2[referenced_cols[0]].nunique()
        else:
            unique_vals1 = table1[dependent_cols].drop_duplicates().shape[0]
            unique_vals2 = table2[referenced_cols].drop_duplicates().shape[0]
        
        # FK should not have more unique values than the referenced table
        return unique_vals1 <= unique_vals2 * 1.1  # Allow 10% tolerance
    
    def print_results(self):
        """Print the detected primary keys and foreign keys."""
        print("=" * 50)
        print("DATABASE SCHEMA ANALYSIS RESULTS")
        print("=" * 50)
        
        print("\n📊 TABLE INFORMATION:")
        for table_name, info in self.table_info.items():
            print(f"  {table_name}: {info['rows']} rows, {info['columns']} columns")
        
        print("\n🔑 PRIMARY KEYS:")
        if self.primary_keys:
            for table_name, pk in self.primary_keys.items():
                print(f"  {table_name}: {pk}")
        else:
            print("  No primary keys detected")
        
        print("\n🔗 FOREIGN KEYS:")
        if self.foreign_keys:
            for table_name, fks in self.foreign_keys.items():
                print(f"  {table_name}:")
                for fk in fks:
                    dependent_cols = fk['dependent'][1]
                    referenced_table, referenced_cols = fk['referenced']
                    print(f"    {dependent_cols} -> {referenced_table}.{referenced_cols} (score: {fk['score']:.3f})")
        else:
            print("  No foreign keys detected")
        
        print("\n📈 STATISTICS:")
        print(f"  Total UCCs found: {sum(len(uccs) for uccs in self.uccs.values())}")
        print(f"  Total INDs found: {len(self.inds)}")
        print(f"  Primary keys detected: {len(self.primary_keys)}")
        print(f"  Foreign key relationships: {sum(len(fks) for fks in self.foreign_keys.values())}")
    
    def save_results(self, output_file: str):
        """Save results to a JSON file."""
        results = {
            'metadata': {
                'csv_directory': str(self.csv_directory),
                'tables_processed': len(self.tables),
                'total_uccs': sum(len(uccs) for uccs in self.uccs.values()),
                'total_inds': len(self.inds),
                'primary_keys_detected': len(self.primary_keys),
                'foreign_keys_detected': sum(len(fks) for fks in self.foreign_keys.values())
            },
            'table_info': self.table_info,
            'primary_keys': self.primary_keys,
            'foreign_keys': {
                table: [
                    {
                        'columns': fk['dependent'][1],
                        'references': {
                            'table': fk['referenced'][0],
                            'columns': fk['referenced'][1]
                        },
                        'score': fk['score']
                    }
                    for fk in fks
                ]
                for table, fks in self.foreign_keys.items()
            },
            'unique_column_combinations': self.uccs
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_sql_schema(self) -> str:
        """Generate SQL CREATE TABLE statements based on detected keys."""
        sql_statements = []
        
        for table_name, table in self.tables.items():
            # Start CREATE TABLE statement
            sql = f"CREATE TABLE {table_name} (\n"
            
            # Add columns
            columns = []
            for col in table.columns:
                dtype = table[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "DECIMAL"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "DATETIME"
                else:
                    sql_type = "VARCHAR(255)"
                
                columns.append(f"  {col} {sql_type}")
            
            sql += ",\n".join(columns)
            
            # Add primary key constraint
            if table_name in self.primary_keys:
                pk_cols = ", ".join(self.primary_keys[table_name])
                sql += f",\n  PRIMARY KEY ({pk_cols})"
            
            sql += "\n);\n"
            sql_statements.append(sql)
        
        # Add foreign key constraints
        for table_name, fks in self.foreign_keys.items():
            for fk in fks:
                dependent_cols = ", ".join(fk['dependent'][1])
                referenced_table, referenced_cols = fk['referenced']
                referenced_cols_str = ", ".join(referenced_cols)
                
                constraint_name = f"FK_{table_name}_{referenced_table}"
                sql = f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                sql += f"FOREIGN KEY ({dependent_cols}) REFERENCES {referenced_table}({referenced_cols_str});\n"
                sql_statements.append(sql)
        
        return "\n".join(sql_statements)


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect primary and foreign keys in CSV database files')
    parser.add_argument('--directory', default="/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/sakila-db", help='Directory containing CSV files')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--sql', '-s', help='Output SQL file for schema')
    parser.add_argument('--encoding', default='utf-8', help='CSV file encoding')
    parser.add_argument('--max-combo', type=int, default=3, help='Maximum column combination size')
    parser.add_argument('--min-rows', type=int, default=2, help='Minimum rows threshold')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = HoPFCSVDetector(
            csv_directory=args.directory,
            encoding=args.encoding,
            max_combination_size=args.max_combo,
            min_rows_threshold=args.min_rows
        )
        
        # Detect keys
        results = detector.detect_keys()
        
        # Print results
        detector.print_results()
        
        # Save results if requested
        if args.output:
            detector.save_results(args.output)
        
        # Generate SQL schema if requested
        if args.sql:
            sql_schema = detector.generate_sql_schema()
            with open(args.sql, 'w') as f:
                f.write(sql_schema)
            logger.info(f"SQL schema saved to {args.sql}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage without command line arguments
    # detector = HoPFCSVDetector("./database_csv_files")
    # results = detector.detect_keys()
    # detector.print_results()
    # detector.save_results("key_detection_results.json")
    
    # Run main function for command line usage
    exit(main())