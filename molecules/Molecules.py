from __future__ import print_function
import os
from utils import copy_dir

class Molecules(object):

    def __init__(self):
        self.build_directories()

    def build_directories(self):
        """
        Builds directories "./images/", "./input_data/", "./output_data/",
	    "./saved_models/", "./raw_data/"
        """
        print("Building Directories...")

        dirs = ["./images/", "./input_data/", "./output_data/", 
                "./saved_models/", "./raw_data/", "./processed_data", "./history"]

        for path in dirs:
            if not os.path.exists(path):
                os.mkdir(path, 0755)

        print("Completed directories creation or if already exist - then checked")

    def input_raw_data(self, paths):
        """
        paths : list
            - list of directories containing raw data to the included.
            - EX) input_raw_data(["trajectory1.xtc", "trajectory2.xtc"])
        Copies raw trajectory data into "./raw_data" directory.
        """
        print("Copying files ...")
        for path in paths:
            if (not os.path.exists(path)):
                raise Exception("Path " + str(path) + " does not exist!")
            copy_dir.recursive_copy_files(path, "./raw_data/")
        print("Files copied.")

    def empty_raw_data(self):
        # Not implemented
        pass
