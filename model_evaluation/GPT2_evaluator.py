import string

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from benckmarks.benchmark import SpatialCommonsenseHeightBenchmark, SpatialCommonsenseSizeBenchmark, \
    SpatialCommonsensePosrelBenchmark, ShapeVLCommonsenseTestBenchmark, \
    MaterialVLCommonsenseTestBenchmark, \
    ColorVLCommonsenseTestBenchmark, WikiShapeVLCommonsenseTestBenchmark, WikiMaterialVLCommonsenseTestBenchmark, \
    WikiColorVLCommonsenseTestBenchmark, SizeSmallerVLCommonsenseTestBenchmark, \
    SizeLargerVLCommonsenseTestBenchmark
from .model_evaluator import ModelEvaluator


class GPT2Evaluator(ModelEvaluator):
    def __init__(self, benchmark):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        super().__init__(model, benchmark)
        self.model.to(self.device)
        self.tokenizer = tokenizer


class GPT2EvaluatorOnHeightCommonsense(GPT2Evaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseHeightBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"Answer this question with yes or no: {question}\nAnswer:"
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True).to(self.device)

                # Generate output
                output = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 5,  # Limit length to encourage a one-word answer
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    temperature=0.7,  # Control randomness
                    top_k=50,  # Use top-k sampling
                    top_p=0.9  # Use nucleus sampling
                )

                # Decode the generated output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

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


class GPT2EvaluatorOnSizeCommonsense(GPT2Evaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseSizeBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"Answer this question with yes or no: {question}\nAnswer:"
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True).to(self.device)

                # Generate output
                output = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 5,  # Limit length to encourage a one-word answer
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    temperature=0.7,  # Control randomness
                    top_k=50,  # Use top-k sampling
                    top_p=0.9  # Use nucleus sampling
                )

                # Decode the generated output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

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


class GPT2EvaluatorOnPosrelCommonsense(GPT2Evaluator):
    def __init__(self):
        benchmark = SpatialCommonsensePosrelBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"Answer this question with yes or no: {question}\nAnswer:"
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True).to(self.device)

                # Generate output
                output = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 5,  # Limit length to encourage a one-word answer
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    temperature=0.7,  # Control randomness
                    top_k=50,  # Use top-k sampling
                    top_p=0.9  # Use nucleus sampling
                )

                # Decode the generated output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

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


# Evaluator for GPT-2 on VL-commonsense
class GPT2VLCommonsenseEvaluator(GPT2Evaluator):
    def __init__(self, benchmark, prompt):
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=1, shuffle=False)
        self.prompt = prompt

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

            # Create a prompt suitable for GPT-2
            prompt = self.prompt.format(subject=subject)
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            # Generate the model output
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 5,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the predicted object
            predicted_text = generated_text[len(prompt):].strip()
            # Remove punctuation from the predicted text
            predicted_text = predicted_text.translate(str.maketrans('', '', string.punctuation))
            # Get the first word as the predicted shape
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


class GPT2VLCommonsenseShapeEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ShapeVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/shape-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseMaterialEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = MaterialVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/material-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseColorEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = ColorVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/color-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseWikiShapeEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiShapeVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-shape-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseWikiMaterialEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiMaterialVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-material-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseWikiColorEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = WikiColorVLCommonsenseTestBenchmark()
        prompt = self.load_prompt_template('./VL-commonsense_preprocessed/wiki-color-prompt-template.txt')
        super().__init__(benchmark, prompt)


class GPT2VLCommonsenseSizeLargerEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SizeLargerVLCommonsenseTestBenchmark()
        prompt = "Answer this question with yes or no: Is {subject} larger than {object}?\nAnswer:"
        super().__init__(benchmark, prompt)

    def evaluate(self):
        count_correct = 0
        for item in self.dataloader:
            subject = item['sub'][0]
            object = item['obj'][0].lower()
            question = self.prompt.format(subject=subject, object=object)

            inputs = self.tokenizer.encode(question, return_tensors="pt", padding=True).to(self.device)

            # Generate output
            output = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + 5,  # Limit length to encourage a one-word answer
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                temperature=0.7,  # Control randomness
                top_k=50,  # Use top-k sampling
                top_p=0.9  # Use nucleus sampling
            )

            # Decode the generated output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

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


class GPT2VLCommonsenseSizeSmallerEvaluator(GPT2VLCommonsenseEvaluator):
    def __init__(self):
        benchmark = SizeSmallerVLCommonsenseTestBenchmark()
        prompt = "Answer this question with yes or no: Is {subject} smaller than {object}?\nAnswer:"
        super().__init__(benchmark, prompt)

    def evaluate(self):
        count_correct = 0
        for item in self.dataloader:
            subject = item['sub'][0]
            object = item['obj'][0].lower()
            question = self.prompt.format(subject=subject, object=object)

            inputs = self.tokenizer.encode(question, return_tensors="pt", padding=True).to(self.device)

            # Generate output
            output = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + 5,  # Limit length to encourage a one-word answer
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                temperature=0.7,  # Control randomness
                top_k=50,  # Use top-k sampling
                top_p=0.9  # Use nucleus sampling
            )

            # Decode the generated output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

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
