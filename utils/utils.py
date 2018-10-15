import pickle


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
