from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from benckmarks.benchmark import SpatialCommonsenseHeightBenchmark, SpatialCommonsenseSizeBenchmark, \
    SpatialCommonsensePosrelBenchmark
from .model_evaluator import ModelEvaluator


class GPT2Evaluator(ModelEvaluator):
    def __init__(self, benchmark):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
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

                correct_label = label == 0

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader)
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

                correct_label = label == 0

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader)
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

                correct_label = label == 0

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader)
        self.write_log()

        return self.benchmark_log
