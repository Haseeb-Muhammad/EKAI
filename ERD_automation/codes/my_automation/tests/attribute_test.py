from my_automation.attribute import Attribute
import pytest

class TestAttribute:
    """Test suite for Attribute class"""
    
    @pytest.mark.parametrize("table_name,attribute_name,values,expected_fullname", [
        ("users", "id", ["1", "2", "3"], "users.id"),
        ("orders", "customer_id", ["100", "101", "102"], "orders.customer_id"),
        ("products", "name", ["Product A", "Product B"], "products.name"),
        ("", "test", ["value"], ".test"),
        ("table", "", ["value"], "table."),
    ])
    def test_fullname_generation(self, table_name, attribute_name, values, expected_fullname):
        """Test full name generation for various table and attribute combinations"""
        attr = Attribute(table_name, attribute_name, values)
        assert attr.fullName == expected_fullname
    
    @pytest.mark.parametrize("attribute_name,suffix_list,expected_result", [
        ("user_id", ["key", "id", "nr", "no"], 1),
        ("customer_key", ["key", "id", "nr", "no"], 1),
        ("order_nr", ["key", "id", "nr", "no"], 1),
        ("invoice_no", ["key", "id", "nr", "no"], 1),
        ("name", ["key", "id", "nr", "no"], 0),
        ("email", ["key", "id", "nr", "no"], 0),
        ("description", ["key", "id", "nr", "no"], 0),
        ("product_identification", ["key", "id", "nr", "no"], 1),  # Contains "id"
    ])
    def test_check_suffix(self, attribute_name, suffix_list, expected_result):
        """Test suffix checking with various attribute names and suffix lists"""
        attr = Attribute("test_table", attribute_name, ["value1", "value2"])
        result = attr.check_suffix(suffix_list)
        assert result == expected_result
    
    @pytest.mark.parametrize("attribute_name,custom_suffix_list,expected_result", [
        ("user_pk", ["pk", "primary"], 1),
        ("order_primary", ["pk", "primary"], 1),
        ("name", ["pk", "primary"], 0),
        ("customer_identifier", ["identifier", "uid"], 1),
    ])
    def test_check_suffix_custom_list(self, attribute_name, custom_suffix_list, expected_result):
        """Test suffix checking with custom suffix lists"""
        attr = Attribute("test_table", attribute_name, ["value1", "value2"])
        result = attr.check_suffix(custom_suffix_list)
        assert result == expected_result
    
    @pytest.mark.parametrize("values,expected_min_length", [
        (["a", "b", "c"], 1),                    # Short values
        (["12345678", "87654321"], 1),           # Exactly 8 characters
        (["123456789", "987654321"], 0.5),       # 9 characters
        (["1234567890", "0987654321"], 0.33),    # 10 characters
        (["very_long_value_here", "another_long_value"], 0.125),  # Long values
    ])
    def test_value_length_scoring(self, values, expected_min_length):
        """Test value length scoring with various value lengths"""
        attr = Attribute("test_table", "test_attr", values)
        # The actual calculation depends on the max length in values
        max_len = max([len(x) for x in values])
        expected_score = 1 / max(1, max_len - 8)
        assert abs(attr.value_length - expected_score) < 0.01
    
    @pytest.mark.parametrize("table_name,attribute_name,values", [
        ("users", "user_id", ["1", "2", "3"]),
        ("orders", "order_key", ["A001", "A002", "A003"]),
        ("products", "product_nr", ["P1", "P2", "P3"]),
        ("customers", "customer_no", ["C001", "C002", "C003"]),
    ])
    
    def test_pk_score_calculation(self, table_name, attribute_name, values):
        """Test primary key score calculation for potential primary key attributes"""
        attr = Attribute(table_name, attribute_name, values)
        
        # PK score should be sum of all components
        expected_score = (attr.uniquness + attr.cardinality + 
                         attr.value_length + attr.position + attr.suffix)
        assert abs(attr.pkScore - expected_score) < 0.01
        
        # For attributes with PK-like suffixes, score should be higher
        assert attr.pkScore > 2.0  # Minimum expected for PK candidates
    
    @pytest.mark.parametrize("attribute_name,values,expected_components", [
        ("user_id", ["1", "2", "3"], {"cardinality": 1, "position": 0, "suffix": 1}),
        ("name", ["Alice", "Bob", "Charlie"], {"cardinality": 1, "position": 0, "suffix": 0}),
        ("product_key", ["KEY1", "KEY2"], {"cardinality": 1, "position": 0, "suffix": 1}),
    ])
    def test_pk_score_components(self, attribute_name, values, expected_components):
        """Test individual components of primary key score"""
        attr = Attribute("test_table", attribute_name, values)
        
        assert attr.cardinality == expected_components["cardinality"]
        assert attr.position == expected_components["position"]
        assert attr.suffix == expected_components["suffix"]
    
    @pytest.mark.parametrize("values", [
        ([]),  # Empty values list
        (["single_value"]),  # Single value
        (["a"] * 1000),  # Many duplicate values
        ([str(i) for i in range(1000)]),  # Many unique values
    ])
    def test_edge_cases(self, values):
        """Test edge cases with various value list configurations"""
        if not values:
            # Empty values should raise an error or handle gracefully
            with pytest.raises((ValueError, ZeroDivisionError)):
                Attribute("test_table", "test_attr", values)
        else:
            attr = Attribute("test_table", "test_attr", values)
            assert attr.fullName == "test_table.test_attr"
            assert 0 <= attr.uniquness <= 1
            assert attr.cardinality == 1
            assert attr.position == 0
            assert attr.suffix in [0, 1]
