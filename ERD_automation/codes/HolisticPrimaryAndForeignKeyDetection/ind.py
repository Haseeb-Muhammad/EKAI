class IND:
    """
    A class representing Inclusion Dependencies between attributes.
    
    Parameters
    ----------
    dependent : Attribute
        The dependent attribute in the inclusion dependency
    reference : Attribute
        The reference attribute in the inclusion dependency
    """
    
    def __init__(self, dependent, reference):
        self.dependent = dependent
        self.reference = reference