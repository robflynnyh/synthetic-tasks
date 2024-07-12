import os

def get_relative_path(path): #get relative path from current file
    return os.path.join(os.path.dirname(__file__), path)