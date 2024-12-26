import string

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from torch.utils.data import DataLoader

from benckmarks.benchmark import SpatialCommonsenseHeightBenchmark, SpatialCommonsenseSizeBenchmark, \
    SpatialCommonsensePosrelBenchmark, WikiColorVLCommonsenseTestBenchmark, WikiMaterialVLCommonsenseTestBenchmark, \
    WikiShapeVLCommonsenseTestBenchmark, ColorVLCommonsenseTestBenchmark, MaterialVLCommonsenseTestBenchmark, \
    ShapeVLCommonsenseTestBenchmark, SizeSmallerVLCommonsenseTestBenchmark, SizeLargerVLCommonsenseTestBenchmark
from .model_evaluator import ModelEvaluator


class FlamingoEvaluator(ModelEvaluator):
    def __init__(self, benchmark):
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            # cache_dir="~/.cache"
        )

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

        super().__init__(model, benchmark)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        model.to(self.device)

        simple_white_image = Image.open('misc/simple-white.png')
        vision_x = [self.image_processor(simple_white_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        self.vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)


class FlamingoEvaluatorOnHeightCommonsense(FlamingoEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseHeightBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)

    def evaluate(self):
        count_correct = 0

        for sample in self.dataloader:
            question, label = sample['question'][0], sample['label'][0]
            prompt = f"<image>ignore the content of image for answering. <|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
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


class FlamingoEvaluatorOnSizeCommonsense(FlamingoEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseSizeBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for sample in self.dataloader:
            question, label = sample['question'][0], sample['label'][0]
            prompt = f"<image>ignore the content of image for answering. <|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
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


class FlamingoEvaluatorOnPosrelCommonsense(FlamingoEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsensePosrelBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for sample in self.dataloader:
            question, label = sample['question'][0], sample['label'][0]
            prompt = f"<image>ignore the content of image for answering. <|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
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


class FlamingoVLCommonsenseEvaluator(FlamingoEvaluator):
    def __init__(self, benchmark, prompt):
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)
        self.prompt = "<image>ignore the content of image for answering. <|endofchunk|>" + prompt

    @staticmethod
    def load_prompt_template(file_path):
        with open(file_path, 'r') as file:
            template = file.read().strip()
            return template

    def evaluate(self):
        count_correct = 0
        for item in self.dataloader:
            subject = item['sub'][0]
            correct_obj = item['obj'][0].lower()

            prompt = self.prompt.format(subject=subject)
            # Encode the prompt
            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [prompt],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
                generated_text = self.model.generate(
                    vision_x=self.vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=20,
                    num_beams=3,
                )

            # Decode the generated text
            generated_text = self.tokenizer.decode(generated_text[0])
            # Extract the predicted object
            predicted_text = generated_text[len(prompt):].strip().rstrip('<|endofchunk|>')
            # Remove punctuation from the predicted text
            predicted_text = predicted_text.translate(str.maketrans('', '', string.punctuation))
            # Get the first word as the predicted object
            predicted_obj = predicted_text.split()[0].lower()
            # Standardize the predicted object
            predicted_obj = self.benchmark.standard_mapping.get(predicted_obj, predicted_obj)
            # Compare the standardized predicted object with the standardized correct object
            if predicted_obj == correct_obj:
                count_correct += 1

        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader.dataset)
        self.write_log()

        return self.benchmark_log


class FlamingoVLCommonsenseShapeEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ShapeVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/shape-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseMaterialEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = MaterialVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/material-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseColorEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ColorVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/color-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiShapeEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiShapeVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-shape-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiMaterialEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiMaterialVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-material-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseWikiColorEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiColorVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-color-prompt-template.txt')
        super().__init__(benchmark, prompt)


class FlamingoVLCommonsenseSizeSmallerEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SizeSmallerVLCommonsenseTestBenchmark()
        prompt = "After the next question comes \"yes\" or \"no\": Is {subject} smaller than {object}?\nAnswer:"
        super().__init__(benchmark, prompt)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for sample in self.dataloader:
            subject = sample['sub'][0]
            object = sample['obj'][0].lower()
            question = self.prompt.format(subject=subject, object=object)

            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [question],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
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
            answer = generated_text[len(question):].strip().lower()

            # Implement specific evaluation logic here
            predicted_label = None
            if "no" in answer:
                predicted_label = False
            elif "yes" in answer:
                predicted_label = True
            else:
                self.benchmark_log["ambiguous_outputs"].append([question, answer])

            correct_label = True
            # The benchmark is size_smaller, so the correct label is always True

            count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader.dataset)
        self.write_log()

        return self.benchmark_log


class FlamingoVLCommonsenseSizeLargerEvaluator(FlamingoVLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SizeLargerVLCommonsenseTestBenchmark()
        prompt = "After the next question comes \"yes\" or \"no\": Is {subject} larger than {object}?\nAnswer:"
        super().__init__(benchmark, prompt)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for sample in self.dataloader:
            subject = sample['sub'][0]
            object = sample['obj'][0].lower()
            question = self.prompt.format(subject=subject, object=object)

            self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
            lang_x = self.tokenizer(
                [question],
                return_tensors="pt",
            ).to(self.device)

            generated_text = ''
            with torch.no_grad():
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
            answer = generated_text[len(question):].strip().lower()

            # Implement specific evaluation logic here
            predicted_label = None
            if "no" in answer:
                predicted_label = False
            elif "yes" in answer:
                predicted_label = True
            else:
                self.benchmark_log["ambiguous_outputs"].append([question, answer])

            correct_label = True
            # The benchmark is size_larger, so the correct label is always True

            count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader.dataset)
        self.write_log()

        return self.benchmark_log
