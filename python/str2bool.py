import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'Yes', 'true', 'True', 't', 'T', 'y', '1'):
        return True
    elif v.lower() in ('no', 'No', 'false', 'False', 'f', 'F', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_float(value):
    try:
        # This will try to convert directly to float or evaluate as a fraction
        return float(eval(value))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Invalid float or expression: %r" % value)

