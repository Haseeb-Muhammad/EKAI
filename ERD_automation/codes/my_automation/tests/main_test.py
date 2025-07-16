from my_automation.main import auto_incremental_pk_pruning
from my_automation.attribute import Attribute
from my_automation.ind import IND
import pytest

class TestAutoIncrementalPKPruning:
    """Test suite for auto_incremental_pk_pruning function"""

    def create_ind(self, dependent_values, reference_values, table_names=None):
        """Helper method to create IND objects for testing"""
        if table_names is None:
            table_names = ("table1", "table2")
            
        dependent = Attribute(table_names[0], "col1", dependent_values)
        reference = Attribute(table_names[1], "col2", reference_values)
        return IND(dependent, reference)

    @pytest.mark.parametrize("test_case", [
        {
            "name": "auto_incremental_sequence",
            "input_values": [("1,2", "1,2,3,4")],
            "expected_count": 0
        },
        {
            "name": "non_consecutive_values",
            "input_values": [("1,3", "1,2,3,4")],
            "expected_count": 1
        },
        {
            "name": "empty_input",
            "input_values": [],
            "expected_count": 0
        },
        {
            "name": "multiple_inds_mixed",
            "input_values": [
                ("1,2", "1,2,3"),
                ("2,4", "1,2,3,4")
            ],
            "expected_count": 1
        },
        {
            "name": "equal_length_sequences",
            "input_values": [("1,2,3", "1,2,3")],
            "expected_count": 0
        }
    ])
    def test_auto_incremental_pk_pruning(self, test_case):
        """Test auto_incremental_pk_pruning with various scenarios"""
        # Arrange
        input_inds = []
        for dep_ref in test_case["input_values"]:
            if len(dep_ref) == 2:
                dependent_vals = dep_ref[0].split(",")
                reference_vals = dep_ref[1].split(",")
                input_inds.append(self.create_ind(dependent_vals, reference_vals))

        # Act
        result = auto_incremental_pk_pruning(input_inds)

        # Assert
        assert len(result) == test_case["expected_count"], \
            f"Test case '{test_case['name']}' failed: expected {test_case['expected_count']} INDs, got {len(result)}"

    @pytest.mark.parametrize("dependent_vals,reference_vals,expected_result", [
        (["1"], ["1", "2", "3"], False),  # Single value subset
        (["4"], ["1", "2", "3"], True),   # Single value not in sequence
        (["1", "2"], ["2", "1", "3"], True),  # Non-sequential order
        ([], [], False),  # Empty values
    ])
    def test_edge_cases(self, dependent_vals, reference_vals, expected_result):
        """Test edge cases for auto_incremental_pk_pruning"""
        # Arrange
        ind = self.create_ind(dependent_vals, reference_vals)
        
        # Act
        result = auto_incremental_pk_pruning([ind])
        
        # Assert
        assert bool(len(result)) == expected_result