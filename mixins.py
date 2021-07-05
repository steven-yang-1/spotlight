class DataPreprocessMixin:
    X = []

    def zero_centered(self):
        self.X = self.X - np.mean(self.X, axis=0, keepdims=True)

    def standardize(self):
        self.X = (self.X - np.mean(self.X, axis=0, keepdims=True)) / np.std(self.X, axis=0, ddof=1)

    def normalize(self):
        _range = np.max(self.X) - np.min(self.X)
        return (self.X - np.min(self.X)) / _range