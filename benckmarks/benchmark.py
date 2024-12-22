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


class VLCommonsenseBenchmarkDataset(Dataset):
    def __init__(self, dataset_file, standard_words_file=None):
        with open(dataset_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        if standard_words_file:
            self.standard_mapping = self.build_standard_mapping(standard_words_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def build_standard_mapping(self, shape_words_file):
        # Build shape mapping
        standard_mapping = {}
        with open(shape_words_file, 'r') as f:
            for line in f:
                line = line.strip().lower()
                if not line or line.startswith('#'):
                    continue  # Skip empty or commented lines
                if ':' in line:
                    variants_part, standard = line.split(':')
                    variants = [v.strip() for v in variants_part.strip().split(',')]
                    standard = standard.strip()
                    for variant in variants:
                        if variant:
                            standard_mapping[variant] = standard
                else:
                    word = line.strip()
                    if word:
                        standard_mapping[word] = word  # Map to itself

        return standard_mapping


class ShapeVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/shape/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/shape-words.txt')
        super().__init__(distribution_file, words_file)


class ShapeVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/shape/single/test.jsonl')
        super().__init__(distribution_file)


class MaterialVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/material/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/material-words.txt')
        super().__init__(distribution_file, words_file)


class MaterialVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/material/single/test.jsonl')
        super().__init__(distribution_file)


class ColorVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/color/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/color-words.txt')
        super().__init__(distribution_file, words_file)


class ColorVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/color/single/test.jsonl')
        super().__init__(distribution_file)


class SizeLargerVLCommonsenseBenchmark(BenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/size_larger/train.jsonl')
        super().__init__(distribution_file)


class SizeLargerVLCommonsenseTestBenchmark(BenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/size_larger/test.jsonl')
        super().__init__(distribution_file)


class SizeSmallerVLCommonsenseBenchmark(BenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/size_smaller/train.jsonl')
        super().__init__(distribution_file)


class SizeSmallerVLCommonsenseTestBenchmark(BenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/size_smaller/test.jsonl')
        super().__init__(distribution_file)


class WikiShapeVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-shape/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/shape-words.txt')
        super().__init__(distribution_file, words_file)


class WikiShapeVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-shape/single/test.jsonl')
        super().__init__(distribution_file)


class WikiMaterialVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-material/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/material-words.txt')
        super().__init__(distribution_file, words_file)


class WikiMaterialVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-material/single/test.jsonl')
        super().__init__(distribution_file)


class WikiColorVLCommonsenseBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-color/single/train.jsonl')
        words_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/words/color-words.txt')
        super().__init__(distribution_file, words_file)


class WikiColorVLCommonsenseTestBenchmark(VLCommonsenseBenchmarkDataset):
    def __init__(self):
        distribution_file = os.path.join(PROJECT_ROOT, 'VL-commonsense/mine-data/db/wiki-color/single/test.jsonl')
        super().__init__(distribution_file)
