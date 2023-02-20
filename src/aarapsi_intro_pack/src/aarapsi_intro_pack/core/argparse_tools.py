import argparse as ap

# https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
def check_positive_int(value):
    error_text = "%s is an invalid positive integer value." % (str(value))
    try:
        ivalue = int(value)
    except:
        raise ap.ArgumentTypeError(error_text)
        
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_positive_float(value):
    error_text = "%s is an invalid positive float value." % (str(value))
    try:
        ivalue = float(value)
    except:
        raise ap.ArgumentTypeError(error_text)
        
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_bool(value):
    error_text = "%s is an invalid boolean." % (str(value))
    if isinstance(value, bool): return value
    if isinstance(value, int): return not value == 0
    if isinstance(value, str): return value.lower() == "true"
    raise Exception(error_text)

def check_positive_two_int_tuple(raw_value):
    error_text = "%s is an invalid positive two-integer tuple." % (str(raw_value))
    str_value = str(raw_value) # force to string
    value_list = str_value.replace(' ', '').replace('(','').replace(')','').split(',')
    if not len(value_list) == 2:
        raise ap.ArgumentTypeError(error_text) 
    if '.' in str(str_value):
        raise ap.ArgumentTypeError(error_text) 
    try:
        ivalue = (int(value_list[0]), int(value_list[1]))
    except:
        raise ap.ArgumentTypeError(error_text)
    if not (ivalue[0] > 0 and ivalue[1] > 0):
        raise ap.ArgumentTypeError(error_text)
    return ivalue