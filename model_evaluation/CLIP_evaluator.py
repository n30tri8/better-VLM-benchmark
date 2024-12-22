import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel

from benckmarks.benchmark import SpatialCommonsenseSizeBenchmark, \
    SpatialCommonsensePosrelBenchmark, ShapeVLCommonsenseBenchmark, MaterialVLCommonsenseBenchmark, \
    ColorVLCommonsenseBenchmark, WikiShapeVLCommonsenseBenchmark, WikiMaterialVLCommonsenseBenchmark, \
    WikiColorVLCommonsenseBenchmark, ShapeVLCommonsenseTestBenchmark, ColorVLCommonsenseTestBenchmark, \
    MaterialVLCommonsenseTestBenchmark, WikiShapeVLCommonsenseTestBenchmark, WikiMaterialVLCommonsenseTestBenchmark, \
    WikiColorVLCommonsenseTestBenchmark, SizeLargerVLCommonsenseBenchmark, SizeLargerVLCommonsenseTestBenchmark, \
    SizeSmallerVLCommonsenseBenchmark, SizeSmallerVLCommonsenseTestBenchmark
from .model_evaluator import ModelEvaluator


class CLIPEvaluator(ModelEvaluator):
    def __init__(self, benchmark):
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        super().__init__(model, benchmark)
        self.tokenizer = tokenizer
        model.to(self.device)


class CLIPVLCommonsenseEvaluator(CLIPEvaluator):
    def __init__(self, train_dataset, test_dataset):
        super().__init__(train_dataset)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.all_objects = self.extract_all_objects()

        self.prompt_templates = []

    def extract_all_objects(self):
        objs = []
        for sample in self.train_dataset.data:
            obj = sample['obj']
            if obj not in objs:
                objs.append(obj)
        return objs

    def get_features(self, input_data, template):
        all_labels = []
        all_input_text = []
        for sample in input_data:
            sub = sample['sub']
            obj = sample['obj']
            if obj not in self.all_objects:
                continue

            all_input_text.append(template.format(subject=sub))
            all_labels.append(self.all_objects.index(obj))
        all_labels = np.array(all_labels)

        with torch.no_grad():
            text_input = self.tokenizer(all_input_text, padding=True, return_tensors="pt").to(self.device)
            model_output = self.model(**text_input)
            all_features = model_output.pooler_output
            all_features = all_features.cpu().numpy()

        return all_features, all_labels

    def evaluate(self):
        # Calculate features
        train_all_temps = []
        test_all_temps = []
        for template in self.prompt_templates:
            train_features_all, train_labels_all = self.get_features(self.train_dataset, template)
            test_features, test_labels = self.get_features(self.test_dataset, template)
            train_all_temps.append((train_features_all, train_labels_all))
            test_all_temps.append((test_features, test_labels))

        acc_all_temps = []
        for i in range(len(self.prompt_templates)):
            train_features = train_all_temps[i][0]
            train_labels = train_all_temps[i][1]
            test_features = test_all_temps[i][0]
            test_labels = test_all_temps[i][1]

            # Perform logistic regression
            classifier = LogisticRegression(random_state=0, penalty='l1', C=0.0001, max_iter=500, solver='saga',
                                            verbose=0)
            classifier.fit(train_features, train_labels)

            # Evaluate using the logistic regression classifier
            predictions = classifier.predict(test_features)
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            acc_all_temps.append(accuracy)

        self.benchmark_log["best_acc"] = round(np.max(acc_all_temps), 2)
        return self.benchmark_log


class CLIPVLCommonsenseShapeEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = ShapeVLCommonsenseBenchmark()
        test_dataset = ShapeVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} can be of shape [obj] .",
                                 "{subject} has shape [obj] .",
                                 "{subject} is of shape [obj] .",
                                 "The shape of {subject} can be [obj] .",
                                 "The shape of the {subject} is [obj] .",
                                 "[obj] {subject} .",
                                 "This is a [obj] {subject} .", ]


class CLIPVLCommonsenseWikiShapeEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = WikiShapeVLCommonsenseBenchmark()
        test_dataset = WikiShapeVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} can be of shape [obj] .",
                                 "{subject} has shape [obj] .",
                                 "{subject} is of shape [obj] .",
                                 "The shape of {subject} can be [obj] .",
                                 "The shape of the {subject} is [obj] .",
                                 "[obj] {subject} .",
                                 "This is a [obj] {subject} .", ]


class CLIPVLCommonsenseColorEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = ColorVLCommonsenseBenchmark()
        test_dataset = ColorVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} can be of color [obj] .", "{subject} has color [obj] .",
                                 "The color of {subject} can be [obj] .", "The color of the {subject} is [obj] .",
                                 "[obj] {subject} .", "This is a [obj] {subject} .", "{subject} is of color [obj] ."]


class CLIPVLCommonsenseWikiColorEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = WikiColorVLCommonsenseBenchmark()
        test_dataset = WikiColorVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} can be of color [obj] .",
                                 "{subject} has color [obj] .",
                                 "{subject} is of color [obj] .",
                                 "The color of {subject} can be [obj] .",
                                 "The color of the {subject} is [obj] .",
                                 "[obj] {subject} .",
                                 "This is a [obj] {subject} .", ]


class CLIPVLCommonsenseMaterialEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = MaterialVLCommonsenseBenchmark()
        test_dataset = MaterialVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} is made of [obj] .",
                                 "{subject} can be made of [obj] .",
                                 "{subject} is made from [obj] .",
                                 "{subject} can be made from [obj] .",
                                 "[obj] {subject} .",
                                 "This is a [obj] {subject} .",
                                 "[obj] is used to make {subject} .", ]


class CLIPVLCommonsenseWikiMaterialEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = WikiMaterialVLCommonsenseBenchmark()
        test_dataset = WikiMaterialVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} is made of [obj] .",
                                 "{subject} can be made of [obj] .",
                                 "{subject} is made from [obj] .",
                                 "The material of {subject} can be [obj] .",
                                 "The material of the {subject} is [obj] .",
                                 "[obj] {subject} .",
                                 "This is a [obj] {subject} .", ]


class CLIPVLCommonsenseSizeSmallerEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = SizeSmallerVLCommonsenseBenchmark()
        test_dataset = SizeSmallerVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} is smaller than [obj] .",
                                 "Ths size of {subject} is smaller than that of [obj] .",
                                 "{subject} can be smaller than [obj] .",
                                 "[obj] is larger than {subject} .",
                                 "Ths size of [obj] is larger than that of {subject} .",
                                 "[obj] can be larger than {subject} .",
                                 "[obj] is bigger than {subject} .",
                                 "{subject} is not as big as [obj] .", ]


class CLIPVLCommonsenseSizeLargerEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = SizeLargerVLCommonsenseBenchmark()
        test_dataset = SizeLargerVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} is larger than [obj] ."
                                 "Ths size of {subject} is larger than that of [obj] ."
                                 "{subject} can be larger than [obj] ."
                                 "[obj] is smaller than {subject} ."
                                 "Ths size of [obj] is smaller than that of {subject} ."
                                 "[obj] can be smaller than {subject} ."
                                 "{subject} is bigger than [obj] ."
                                 "[obj] is not as big as {subject} ."]


class CLIPEvaluatorOnSizeCommonsense(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseSizeBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"<image>ignore the content of image for answering<|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

                self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
                lang_x = self.tokenizer(
                    [prompt],
                    return_tensors="pt",
                )

                generated_text = self.model.generate(
                    vision_x=self.vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=20,
                    num_beams=3,
                )

                # Decode the generated output
                generated_text = self.tokenizer.decode(generated_text[0])

                # Extract the answer from the generated text
                answer = generated_text[len(prompt):].strip().lower()

                # Implement specific evaluation logic here
                predicted_label = None
                if "no" in answer:
                    predicted_label = False
                elif "yes" in answer:
                    predicted_label = True
                else:
                    self.benchmark_log["ambiguous_outputs"].append([question, answer])

                correct_label = False if label == 0 else True

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader.dataset)
        self.write_log()

        return self.benchmark_log


class CLIPEvaluatorOnPosrelCommonsense(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsensePosrelBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"<image>ignore the content of image for answering<|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

                self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
                lang_x = self.tokenizer(
                    [prompt],
                    return_tensors="pt",
                )

                generated_text = self.model.generate(
                    vision_x=self.vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=20,
                    num_beams=3,
                )

                # Decode the generated output
                generated_text = self.tokenizer.decode(generated_text[0])

                # Extract the answer from the generated text
                answer = generated_text[len(prompt):].strip().lower()

                # Implement specific evaluation logic here
                predicted_label = None
                if "no" in answer:
                    predicted_label = False
                elif "yes" in answer:
                    predicted_label = True
                else:
                    self.benchmark_log["ambiguous_outputs"].append([question, answer])

                correct_label = False if label == 0 else True

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader.dataset)
        self.write_log()

        return self.benchmark_log
