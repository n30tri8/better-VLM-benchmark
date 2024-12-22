import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel

from benckmarks.benchmark import SpatialCommonsenseSizeBenchmark, \
    SpatialCommonsensePosrelBenchmark, ShapeVLCommonsenseBenchmark, MaterialVLCommonsenseBenchmark, \
    ColorVLCommonsenseBenchmark, WikiShapeVLCommonsenseBenchmark, WikiMaterialVLCommonsenseBenchmark, \
    WikiColorVLCommonsenseBenchmark, ShapeVLCommonsenseTestBenchmark, ColorVLCommonsenseTestBenchmark, \
    MaterialVLCommonsenseTestBenchmark
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
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=2000, verbose=0)
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


class CLIPVLCommonsenseColorEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        train_dataset = ColorVLCommonsenseBenchmark()
        test_dataset = ColorVLCommonsenseTestBenchmark()
        super().__init__(train_dataset, test_dataset)
        self.prompt_templates = ["{subject} can be of color [obj] .", "{subject} has color [obj] .",
                                 "The color of {subject} can be [obj] .", "The color of the {subject} is [obj] .",
                                 "[obj] {subject} .", "This is a [obj] {subject} .", "{subject} is of color [obj] ."]


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


# class CLIPVLCommonsenseEvaluator(CLIPVLCommonsenseEvaluator):
#     def __init__(self, benchmark, prompt):
#         super().__init__(benchmark)
#         self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)
#         self.prompt = "<image>ignore the content of image for answering<|endofchunk|>" + prompt
#
#     def evaluate(self):
#         count_correct = 0
#         for item in self.dataloader:
#             subject = item['sub'][0]
#             correct_obj = item['obj'][0].lower()
#
#             # Create a prompt suitable for GPT-2
#             prompt = self.prompt.format(subject=subject)
#             # Encode the prompt
#             self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
#             lang_x = self.tokenizer(
#                 [prompt],
#                 return_tensors="pt",
#             )
#
#             generated_text = self.model.generate(
#                 vision_x=self.vision_x,
#                 lang_x=lang_x["input_ids"],
#                 attention_mask=lang_x["attention_mask"],
#                 max_new_tokens=20,
#                 num_beams=3,
#             )
#
#             # Decode the generated text
#             generated_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
#             # Extract the predicted object
#             predicted_text = generated_text[len(prompt):].strip().lower()
#             # Remove punctuation from the predicted text
#             predicted_text = predicted_text.translate(str.maketrans('', '', string.punctuation))
#             # Get the first word as the predicted shape
#             predicted_obj = predicted_text.split()[0].lower()
#             # Standardize the predicted object
#             predicted_obj = self.benchmark.standard_mapping.get(predicted_obj, predicted_obj)
#             # Compare the standardized predicted object with the standardized correct object
#             if predicted_obj == correct_obj:
#                 count_correct += 1
#
#         self.benchmark_log["correct"] = count_correct
#         self.benchmark_log["total"] = len(self.dataloader.dataset)
#         self.write_log()
#
#         return self.benchmark_log


class FlamingoVLCommonsenseShapeEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ShapeVLCommonsenseBenchmark()
        prompt = "In one word, the typical shape of a {subject} is a"
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseMaterialEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = MaterialVLCommonsenseBenchmark()
        prompt = "In one word, the typical material of a {subject} is"
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseColorEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ColorVLCommonsenseBenchmark()
        prompt = "In one word, the typical color of a {subject} is"
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiShapeEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiShapeVLCommonsenseBenchmark()
        prompt = "In one word, the typical shape of a {subject} is a"
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiMaterialEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiMaterialVLCommonsenseBenchmark()
        prompt = "In one word, the typical material of a {subject} is"
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiColorEvaluator(CLIPVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiColorVLCommonsenseBenchmark()
        prompt = "In one word, the typical color of a {subject} is"
        super().__init__(benchmark, prompt)
