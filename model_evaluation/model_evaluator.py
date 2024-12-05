import torch


class ModelEvaluator:
    def __init__(self, model, benchmark):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.benchmark = benchmark
        self.benchmark_log = {"correct": 0, "total": 0, "ambiguous_outputs": []}

    def evaluate(self, benchmark):
        pass

    def write_log(self):
        with open(f'classification-logs/{self.__class__.__name__}.txt', 'w') as f:
            f.write(str(self.benchmark_log))
            f.close()
