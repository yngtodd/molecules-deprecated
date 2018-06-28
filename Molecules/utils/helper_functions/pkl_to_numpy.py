import gzip, pickle
from tempfile import TemporaryFile

def convert(path):
    """
    Builds a file containing a numpy array with a pkl file specified by path. 
    """
    with gzip.open(path, 'rb'):
        data, label = pickle.load(f)


   f_data = open("./data", 'w')
   f_data
