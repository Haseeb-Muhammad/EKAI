class Database:
    """
    Class to store information about a database, including its directory and associated file paths.

    Attributes
    ----------
    database_dir : str
        Path to the main directory of the database which contain csv files.
    ind_path : str
        Path to the file storing inclusion dependency in the format.
    gt_path : str
        Path to the ground truth file of the database.
    """

    def __init__(self, database_dir: str, ind_path: str, gt_path: str) -> None:
        self.database_dir = database_dir
        self.ind_path = ind_path
        self.gt_path = gt_path