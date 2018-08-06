import os, shutil;

# Directory paths for extract_native-contact
path_1 = "./native-contact/"
path_2 = "./native-contact/raw/"
path_3 = "./native-contact/data/"

# EFFECTS: Deletes raw and data directories and leaves native-contact directory empty.
def empty_directory(dir_path):
    print "Emptying directory ..."
    for root, dirs, files in os.walk(dir_path):
	for f in files:
	    os.unlink(os.path.join(root, f))
	for d in dirs:
	    shutil.rmtree(os.path.join(root, d))
    print "Empty!"
