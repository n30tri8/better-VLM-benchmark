import torch


class ModelEvaluator:
    def __init__(self, model, benchmark):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.benchmark = benchmark

    def evaluate(self, benchmark):
        pass


