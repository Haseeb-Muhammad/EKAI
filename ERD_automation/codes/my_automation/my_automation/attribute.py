from datasketch import HyperLogLog


class Attribute:
    """
    A class representing a database table attribute with metadata for primary key detection.
    
    Parameters
    ----------
    table_name : str
        Name of the database table
    attribute_name : str
        Name of the attribute/column
    values : list
        List of values in the attribute
        
    Attributes
    ----------
    fullName : str
        Fully qualified name (table_name.attribute_name)
    uniquness : float
        Estimated uniqueness score using HyperLogLog
    cardinality : int
        Cardinality score (default 1)
    value_length : float
        Score based on value lengths
    position : float 
        Score based on column position 
    suffix : int
        Score based on common primary key suffixes
    pkScore : float
        Combined score for primary key likelihood
    
    Methods
    -------
    estUniqueness()
        Estimates uniqueness of values using HyperLogLog
    check_suffix(suffix_list)
        Checks if attribute name contains common primary key suffixes
    """

    def __init__(self, table_name, attribute_name, values):
        self.table_name = table_name
        self.attribute_name = attribute_name
        self.values = values
        self.fullName=f"{self.table_name}.{self.attribute_name}"

        self.uniquness = self.estUniqueness()
        self.cardinality=1
        self.value_length = 1/max(1, max([len(x) for x in values]) - 8) #8 is a hyper-parameter which penalties primary key candidates which has a value with length > 8
        self.position = 0
        self.suffix = self.check_suffix()

        self.pkScore = 0
        self.pkScore += self.uniquness
        self.pkScore += self.cardinality
        self.pkScore += self.value_length
        self.pkScore += self.position
        self.pkScore += self.suffix

    def estUniqueness(self):
        """
        Estimate uniqueness of attribute values using HyperLogLog algorithm.
        
        Returns
        -------
        float
            Ratio of unique values to total values
        """

        #Dealing with all null values atribute
        if len(self.values) == 0:
            return 0

        hll = HyperLogLog()
        total = 0
        
        for value in self.values:
            hll.update(str(value).encode('utf8'))
            total +=1
        
        return min(1, hll.count() / total)
    
    def check_suffix(self, suffix_list=["key", 'id', 'nr', 'no']):
        """
        Check if attribute name contains common primary key suffixes.
        
        Parameters
        ----------
        suffix_list : list of str, optional
            List of common primary key suffix strings to check
            
        Returns
        -------
        int
            1 if suffix found, 0 otherwise
        """
        for suffix in suffix_list:
            if suffix in self.attribute_name:
                return 1
        return 0
