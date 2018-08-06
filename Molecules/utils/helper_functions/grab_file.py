

def grab_file(file_name=None):
    """
    Returns the name of the file but not the entire path as well as strips the .
    EX) 'home/molecules/layer_output_conv1_th.out' -> 'layer_output_conv1_th.'
    """
     if(file_name == None):
        raise ValueError("Must input file_name as parameter.")
     if (not os.path.exists(file_name)):
        raise Exception("Path " + str(file_name) + " does not exist!")

    if file_name[-1] == '/':
        file_name = file_name[:-1]
    i = 0
    i_last = 0
    for c in file_name:
        if c == '/':
            i_last = i
        i += 1
    new_file = file_name[i_last + 1:-3]
    return new_file