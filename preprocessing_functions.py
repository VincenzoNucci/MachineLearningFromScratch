import numpy as np

class StandardScaler:
    def fit(self, inputs):
        return (inputs - np.mean(inputs)) / np.std(inputs)

class Normalizer:
    def fit(self, inputs):
        return (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))