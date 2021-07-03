import pickle


class DataIO:
    @staticmethod
    def export_to(model, file_name):
        with open(file_name, 'wb') as file_to_write:
            pickle.dump(model, file_to_write)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as file_to_read:
            model = pickle.load(file_to_read)
        return model
