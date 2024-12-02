import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

from benckmarks.benchmark import SpatialCommonsenseHeightBenchmark
from .model_evaluator import ModelEvaluator


class GITEvaluator(ModelEvaluator):
    def __init__(self, benchmark):
        processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa").to(self.device)
        del model.git.image_encoder
        del model.git.visual_projection
        model.git.encoder.layer[0].attention.self.image_patch_tokens = 0
        super().__init__(model, benchmark)
        self.processor = processor


class GITEvaluatorOnHeightCommonsense(GITEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseHeightBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"{question}\nAnswer:"
                input_ids = self.processor(text=prompt, add_special_tokens=False).input_ids
                input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

                # Generate output
                generated_ids = self.model.generate(pixel_values=None, input_ids=input_ids, max_length=100)

                # Decode the generated output
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                print(generated_text)
                break

                # Extract the answer from the generated text
                answer = generated_text[len(prompt):].strip().lower()

                # Implement specific evaluation logic here
                predicted_label = None
                if "no" in answer:
                    predicted_label = False
                elif "yes" in answer:
                    predicted_label = True

                correct_label = label == 0

                count_correct += (1 if predicted_label == correct_label else 0)
        return count_correct / len(self.dataloader)
