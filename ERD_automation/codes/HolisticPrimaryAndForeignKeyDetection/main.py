import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from itertools import combinations
from collections import defaultdict

class HoPFDetector:
    """
    Holistic Primary Key and Foreign Key Detection (HoPF) Algorithm
    
    This class implements the HoPF algorithm for automatically detecting
    primary keys and foreign keys in a collection of pandas DataFrames.
    """
    
    def __init__(self, tables: List[pd.DataFrame], table_names: List[str] = None):
        """
        Initialize the HoPF detector with a list of DataFrames.
        
        Args:
            tables: List of pandas DataFrames representing database tables
            table_names: Optional list of table names (defaults to Table_0, Table_1, etc.)
        """
        self.tables = tables
        self.table_names = table_names or [f"Table_{i}" for i in range(len(tables))]
        self.uccs = {}  # Unique Column Combinations
        self.inds = {}  # Inclusion Dependencies
        self.primary_keys = {}
        self.foreign_keys = {}
        
    def detect_keys(self) -> Dict[str, Dict]:
        """
        Main method to detect primary keys and foreign keys.
        
        Returns:
            Dictionary containing detected primary keys and foreign keys
        """
        # Step 1: Discover UCCs for each table
        self._discover_uccs()
        
        # Step 2: Discover INDs between tables
        self._discover_inds()
        
        # Step 3: Score and select primary keys
        self._select_primary_keys()
        
        # Step 4: Score and select foreign keys
        self._select_foreign_keys()
        
        return {
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys
        }
    
    def _discover_uccs(self):
        """
        Discover Unique Column Combinations (UCCs) for each table.
        UCCs are sets of columns that uniquely identify rows.
        """
        for i, (table, table_name) in enumerate(zip(self.tables, self.table_names)):
            uccs = []
            columns = list(table.columns)
            
            # Check single columns first
            for col in columns:
                if self._is_unique_combination(table, [col]):
                    uccs.append([col])
            
            # Check combinations of 2 columns
            for combo in combinations(columns, 2):
                if self._is_unique_combination(table, list(combo)):
                    # Only add if not a superset of existing UCC
                    if not self._is_superset_of_existing_ucc(list(combo), uccs):
                        uccs.append(list(combo))
            
            # For larger combinations, we limit to 3 columns for performance
            # In practice, most primary keys are 1-2 columns
            for combo in combinations(columns, 3):
                if self._is_unique_combination(table, list(combo)):
                    if not self._is_superset_of_existing_ucc(list(combo), uccs):
                        uccs.append(list(combo))
            
            self.uccs[table_name] = uccs
    
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
        """
        Discover Inclusion Dependencies (INDs) between tables.
        An IND exists if values in columns of one table are included in another.
        """
        inds = []
        
        for i, table1_name in enumerate(self.table_names):
            table1 = self.tables[i]
            for j, table2_name in enumerate(self.table_names):
                if i == j:
                    continue
                
                table2 = self.tables[j]
                
                # Check single column INDs
                for col1 in table1.columns:
                    for col2 in table2.columns:
                        if self._is_inclusion_dependency(table1, [col1], table2, [col2]):
                            inds.append({
                                'dependent': (table1_name, [col1]),
                                'referenced': (table2_name, [col2]),
                                'score': self._calculate_ind_score(table1, [col1], table2, [col2])
                            })
                
                # Check multi-column INDs (limited to 2 columns for performance)
                for col1_combo in combinations(table1.columns, 2):
                    for col2_combo in combinations(table2.columns, 2):
                        if self._is_inclusion_dependency(table1, list(col1_combo), 
                                                      table2, list(col2_combo)):
                            inds.append({
                                'dependent': (table1_name, list(col1_combo)),
                                'referenced': (table2_name, list(col2_combo)),
                                'score': self._calculate_ind_score(table1, list(col1_combo), 
                                                                 table2, list(col2_combo))
                            })
        
        self.inds = inds
    
    def _is_inclusion_dependency(self, table1: pd.DataFrame, cols1: List[str], 
                               table2: pd.DataFrame, cols2: List[str]) -> bool:
        """Check if columns in table1 are included in columns of table2."""
        if len(cols1) != len(cols2):
            return False
        
        # Get unique values from both tables (excluding nulls)
        values1 = set()
        values2 = set()
        
        for _, row in table1[cols1].dropna().iterrows():
            if len(cols1) == 1:
                values1.add(row[cols1[0]])
            else:
                values1.add(tuple(row[cols1]))
        
        for _, row in table2[cols2].dropna().iterrows():
            if len(cols2) == 1:
                values2.add(row[cols2[0]])
            else:
                values2.add(tuple(row[cols2]))
        
        if len(values1) == 0:
            return False
        
        # Check if all values in table1 are included in table2
        return values1.issubset(values2)
    
    def _calculate_ind_score(self, table1: pd.DataFrame, cols1: List[str], 
                           table2: pd.DataFrame, cols2: List[str]) -> float:
        """Calculate inclusion dependency score based on coverage and other factors."""
        # Coverage: ratio of matching values
        values1 = set()
        values2 = set()
        
        for _, row in table1[cols1].dropna().iterrows():
            if len(cols1) == 1:
                values1.add(row[cols1[0]])
            else:
                values1.add(tuple(row[cols1]))
        
        for _, row in table2[cols2].dropna().iterrows():
            if len(cols2) == 1:
                values2.add(row[cols2[0]])
            else:
                values2.add(tuple(row[cols2]))
        
        if len(values1) == 0:
            return 0.0
        
        coverage = len(values1.intersection(values2)) / len(values1)
        
        # Bonus for exact inclusion
        inclusion_bonus = 1.0 if values1.issubset(values2) else 0.0
        
        # Penalty for column name dissimilarity (simple heuristic)
        name_similarity = self._calculate_name_similarity(cols1, cols2)
        
        return coverage * 0.7 + inclusion_bonus * 0.2 + name_similarity * 0.1
    
    def _calculate_name_similarity(self, cols1: List[str], cols2: List[str]) -> float:
        """Calculate name similarity between column sets."""
        if len(cols1) != len(cols2):
            return 0.0
        
        similarity = 0.0
        for c1, c2 in zip(cols1, cols2):
            if c1.lower() == c2.lower():
                similarity += 1.0
            elif c1.lower() in c2.lower() or c2.lower() in c1.lower():
                similarity += 0.5
        
        return similarity / len(cols1)
    
    def _select_primary_keys(self):
        """Select primary keys from UCCs using scoring function."""
        for table_name, uccs in self.uccs.items():
            if not uccs:
                continue
            
            # Score each UCC
            scored_uccs = []
            table = self.tables[self.table_names.index(table_name)]
            
            for ucc in uccs:
                score = self._calculate_pk_score(table, ucc, table_name)
                scored_uccs.append((ucc, score))
            
            # Sort by score (descending) and select the best one
            scored_uccs.sort(key=lambda x: x[1], reverse=True)
            
            if scored_uccs:
                self.primary_keys[table_name] = scored_uccs[0][0]
    
    def _calculate_pk_score(self, table: pd.DataFrame, ucc: List[str], table_name: str) -> float:
        """Calculate primary key score for a UCC."""
        # Prefer shorter keys
        length_penalty = 1.0 / len(ucc)
        
        # Prefer keys with 'id' in name
        name_bonus = 0.0
        for col in ucc:
            if 'id' in col.lower():
                name_bonus += 0.3
        
        # Prefer keys with no nulls
        null_penalty = 0.0
        for col in ucc:
            if col in table.columns:
                null_ratio = table[col].isnull().sum() / len(table)
                null_penalty += null_ratio
        
        null_penalty = null_penalty / len(ucc)
        
        # Prefer keys that are referenced by other tables (foreign key bonus)
        fk_bonus = 0.0
        for ind in self.inds:
            if ind['referenced'][0] == table_name and set(ind['referenced'][1]) == set(ucc):
                fk_bonus += 0.2
        
        return length_penalty * 0.4 + name_bonus * 0.2 + (1 - null_penalty) * 0.2 + fk_bonus * 0.2
    
    def _select_foreign_keys(self):
        """Select foreign keys from INDs using scoring function."""
        # Sort INDs by score
        sorted_inds = sorted(self.inds, key=lambda x: x['score'], reverse=True)
        
        selected_fks = []
        
        for ind in sorted_inds:
            dependent_table, dependent_cols = ind['dependent']
            referenced_table, referenced_cols = ind['referenced']
            
            # Check if referenced columns form a primary key
            if (referenced_table in self.primary_keys and 
                set(referenced_cols) == set(self.primary_keys[referenced_table])):
                
                # Additional checks for foreign key validity
                if self._is_valid_foreign_key(ind):
                    selected_fks.append(ind)
        
        # Group by dependent table
        fk_by_table = defaultdict(list)
        for fk in selected_fks:
            fk_by_table[fk['dependent'][0]].append(fk)
        
        self.foreign_keys = dict(fk_by_table)
    
    def _is_valid_foreign_key(self, ind: Dict) -> bool:
        """Additional validation for foreign key candidates."""
        dependent_table, dependent_cols = ind['dependent']
        referenced_table, referenced_cols = ind['referenced']
        
        # Don't allow self-referencing for now (can be extended)
        if dependent_table == referenced_table:
            return False
        
        # Check if the score is above threshold
        return ind['score'] > 0.5
    
    def print_results(self):
        """Print the detected primary keys and foreign keys."""
        print("=== PRIMARY KEYS ===")
        for table_name, pk in self.primary_keys.items():
            print(f"{table_name}: {pk}")
        
        print("\n=== FOREIGN KEYS ===")
        for table_name, fks in self.foreign_keys.items():
            print(f"{table_name}:")
            for fk in fks:
                dependent_cols = fk['dependent'][1]
                referenced_table, referenced_cols = fk['referenced']
                print(f"  {dependent_cols} -> {referenced_table}.{referenced_cols} (score: {fk['score']:.3f})")
    
    def get_results_dict(self) -> Dict:
        """Return results as a structured dictionary."""
        return {
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
            }
        }


# Example usage
if __name__ == "__main__":
    # Create sample tables
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com', 'eve@example.com']
    })
    
    orders = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105],
        'customer_id': [1, 2, 1, 3, 2],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'total': [100.50, 75.25, 150.00, 200.75, 89.99]
    })
    
    products = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5],
        'name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'price': [999.99, 25.99, 79.99, 299.99, 149.99]
    })
    
    order_items = pd.DataFrame({
        'order_id': [101, 101, 102, 103, 104, 105],
        'product_id': [1, 2, 3, 1, 4, 5],
        'quantity': [1, 1, 1, 1, 1, 1],
        'price': [999.99, 25.99, 79.99, 999.99, 299.99, 149.99]
    })
    
    # Initialize detector
    tables = [customers, orders, products, order_items]
    table_names = ['customers', 'orders', 'products', 'order_items']
    
    detector = HoPFDetector(tables, table_names)
    
    # Detect keys
    results = detector.detect_keys()
    
    # Print results
    detector.print_results()
    
    # Get structured results
    structured_results = detector.get_results_dict()
    print("\n=== STRUCTURED RESULTS ===")
    import json
    print(json.dumps(structured_results, indent=2))