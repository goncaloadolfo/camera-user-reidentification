import pickle


class Pickle_module:
    #"../../../Dataset/Pickles/result.p"
    def __init__(self,path):
        self.__path = path


    def dump(self,data):
        # Store data (serialize)
        pickle.dump(data, open(self.__path,'wb'))

    def load(self):
        # Load data (deserialize)
        with open(self.__path, 'rb') as f:
            data = pickle.load(f)
            return data

