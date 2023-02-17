#!/usr/bin/env python3

import os
import numpy as np


def scan_directory(path):
    dir_scan = os.scandir(path) # https://docs.python.org/3/library/os.html#os.scandir
    fil_names = []
    dir_names = []
    file_exts = []
    for entry in dir_scan:
        if entry.is_file():
            if entry.name.startswith('.'):
                continue
            fil_names.append(entry.name)
            file_exts.append(os.path.splitext(entry)[-1].lower())
        elif entry.is_dir(): 
            dir_names.append(entry.name)
        else: raise Exception("Unknown file type detected.")
    return fil_names, dir_names, np.unique(file_exts)

def check_dir_type(path, filetype=None, alltype=False):
    fs, ds, exts = scan_directory(path)
    if filetype == '': filetype = None
    if (len(fs) > 0) and not (filetype is None): # contains a file and we want a file
        if (filetype in exts): # we want a file and it exists
            if (alltype == False): # we don't care if there are other file types present:
                return True
            elif (len(exts) == 1): # there is only one and it is only what we want
                return True
        return False # we wanted a file and it didn't exist, or there were multiple types and we didn't want that
    if (len(ds) > 0) and (filetype is None): # contains a folder and we only want folders
        return True
    return False # either: 
                 # 1. the directory has no files but we want a filetype
                 # 2. the directory has files but we want no files
                 # 3. the directory has no files and we want no files (good), but it has no folders either (and we want folders)

def check_structure(root, ft, at=False, skip=[]):
    if not check_dir_type(root): return False # root must be only directories
    for dir in os.scandir(root):
        if (dir.name in skip) or (dir.path in skip): continue
        if not check_dir_type(dir.path, filetype=ft, alltype=at): return False
    return True