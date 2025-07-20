from fuzzywuzzy import fuzz


class IND:
    """
    Represents an Inclusion Dependency (IND) between two attributes.

    Attributes:
        dependent: The dependent attribute in the IND relationship.
        reference: The reference attribute in the IND relationship.
        name_similarity: A method to calculate the similarity between the names of the dependent and reference attributes.

    Methods:
        __init__(dependent, reference):
            Initializes the IND object with the given dependent and reference attributes.

        _calculate_name_similarity():
            Calculates the similarity score between the attribute names of the dependent and reference attributes using fuzzy string matching.
            Returns:
                float: The similarity score as a value between 0 and 1.
    """
    def __init__(self, dependent, reference):
        self.dependent = dependent
        self.reference = reference
        self.name_similarity = self.calculate_name_similarity
        self.candidate_confirmation = False

    def calculate_name_similarity(self): 
        """
        Calculates the similarity between the attribute names of the dependent and reference objects using the partial ratio method from the `fuzz` library.

        Returns:
            float: The similarity score as a float between 0 and 1.
        """
        similarity = fuzz.partial_ratio(self.dependent.attribute_name, self.reference.attribute_name) / 100
        if similarity > 80:
            self.candidate_confirmation == True
        else:
            pass

        return similarity /100
