import _pickle as pickle
from datetime import datetime

import bcolz
import jsonpickle
import numpy as np
from jsonpickle.ext import numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


def get_idx(class_name, le):
    class_idx = np.where(le.classes_ == class_name)[0]
    if len(class_idx) == 0:
        raise ValueError("Genre should be one of {}".format(le.classes_))
    return class_idx[0]


def generate_run_id():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def save(obj, file_name):
    if file_name.endswith('.json'):
        with open(file_name, 'w') as f:
            f.write(jsonpickle.dumps(obj))
        return

    if isinstance(obj, np.ndarray):
        np.save(file_name, obj)
        return

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(file_name):
    if file_name.endswith('.json'):
        with open(file_name, 'r') as f:
            return jsonpickle.loads(f.read())

    if file_name.endswith('.npy'):
        return np.load(file_name)

    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def bgr2rgb(img):
    """Converts an RGB image to BGR and vice versa
    """
    return img[..., ::-1]
