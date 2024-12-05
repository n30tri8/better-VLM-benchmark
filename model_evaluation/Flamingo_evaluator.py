import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from torch.utils.data import DataLoader

from benckmarks.benchmark import SpatialCommonsenseHeightBenchmark
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


class FlamingoEvaluatorOnHeightCommonsense(FlamingoEvaluator):
    def __init__(self):
        benchmark = SpatialCommonsenseHeightBenchmark()
        super().__init__(benchmark)
        self.dataloader = DataLoader(benchmark, batch_size=32, shuffle=False)

    def evaluate(self):
        count_correct = 0
        simple_white = Image.open('simple-white.png')
        vision_x = [self.image_processor(simple_white).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        for batch in self.dataloader:
            for question, label in zip(batch['question'], batch['label']):
                prompt = f"<image>ignore the content of image for answering<|endofchunk|>After the next question comes \"yes\" or \"no\": {question}\nAnswer:"

                self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
                lang_x = self.tokenizer(
                    [prompt],
                    return_tensors="pt",
                )

                generated_text = self.model.generate(
                    vision_x=vision_x,
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

                correct_label = label == 0

                count_correct += (1 if predicted_label == correct_label else 0)
        self.benchmark_log["correct"] = count_correct
        self.benchmark_log["total"] = len(self.dataloader)
        self.write_log()

        return self.benchmark_log
