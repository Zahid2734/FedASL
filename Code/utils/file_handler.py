import pickle

def save_file(file_name,data):
    open_file = open(file_name, "wb")
    pickle.dump(data, open_file)
    open_file.close()

def open_file(file_name):
    open_file = open(file_name, "rb")
    Dataset = pickle.load(open_file)
    open_file.close()
    return Dataset