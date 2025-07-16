from my_automation.attribute import Attribute
from my_automation.ind import IND
import pytest

class TestIND:
    """Test suite for IND class"""
    
    @pytest.mark.parametrize("dependent_name,reference_name", [
        ("user_id", "customer_id"),
        ("name", "full_name"),
        ("email", "email_address"),
        ("phone", "phone_number"),
        ("address", "home_address"),
    ])
    def test_ind_initialization(self, dependent_name, reference_name):
        """Test IND object initialization with various attribute combinations"""
        dependent = Attribute(dependent_name, attribute_name="", values = [" "])
        reference = Attribute(reference_name, attribute_name="", values = [" "])
        
        ind = IND(dependent, reference)
        
        assert ind.dependent == dependent
        assert ind.reference == reference
        assert callable(ind.name_similarity)
    
    @pytest.mark.parametrize("dependent_name,reference_name", [
        ("user_id", "user_id"),
        ("name", "name"),
        ("email", "email"),
    ])
    def test_identical_attributes(self, dependent_name, reference_name):
        """Test IND with identical dependent and reference attributes"""
        dependent = Attribute(dependent_name, attribute_name="", values = [" "])
        reference = Attribute(reference_name, attribute_name="", values = [" "])
        
        ind = IND(dependent, reference)
        similarity = ind.calculate_name_similarity()
        
        assert similarity == 1.0

