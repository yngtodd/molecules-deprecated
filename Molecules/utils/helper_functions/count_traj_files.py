from __future__ import print_function
import glob
def count_traj_files(path, extension):
    """
    path : string
        Path of directory containing trajectory files.
    extension : string
        File extension type for trajectory files.
        EX) 'dcd', 'xtc', ...
    """
    return len(glob.glob1(path,"*."+extension))




# USAGE:

n = count_traj_files("./", 'xtc')
print(n)