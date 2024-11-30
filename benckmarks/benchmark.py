import json
import os

from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BenchmarkDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SpatialCommonsenseHeightBenchmark(BenchmarkDataset):
    def __init__(self):
        super().__init__(os.path.join(PROJECT_ROOT, 'spatial-commonsense/data/height/data.json'))


class SpatialCommonsenseSizeBenchmark(BenchmarkDataset):
    def __init__(self):
        super().__init__(os.path.join(PROJECT_ROOT, 'spatial-commonsense/data/size/data.json'))


class SpatialCommonsensePosrelBenchmark(BenchmarkDataset):
    def __init__(self):
        super().__init__(os.path.join(PROJECT_ROOT, 'spatial-commonsense/data/posrel/data_qa.json'))
