import gc

import torch

from model_evaluation.CLIP_evaluator import CLIPVLCommonsenseShapeEvaluator, CLIPVLCommonsenseColorEvaluator, \
    CLIPVLCommonsenseMaterialEvaluator, CLIPVLCommonsenseWikiShapeEvaluator, CLIPVLCommonsenseWikiColorEvaluator, \
    CLIPVLCommonsenseWikiMaterialEvaluator, CLIPVLCommonsenseSizeSmallerEvaluator, CLIPVLCommonsenseSizeLargerEvaluator, \
    CLIPSpatialCommonsenseSizeEvaluator, CLIPSpatialCommonsenseHeightEvaluator
from model_evaluation.Flamingo_evaluator import FlamingoEvaluatorOnHeightCommonsense, \
    FlamingoEvaluatorOnSizeCommonsense, FlamingoVLCommonsenseColorEvaluator, FlamingoVLCommonsenseMaterialEvaluator, \
    FlamingoVLCommonsenseWikiShapeEvaluator, FlamingoVLCommonsenseWikiColorEvaluator, \
    FlamingoVLCommonsenseWikiMaterialEvaluator, FlamingoEvaluatorOnPosrelCommonsense, \
    FlamingoVLCommonsenseShapeEvaluator, FlamingoVLCommonsenseSizeLargerEvaluator, \
    FlamingoVLCommonsenseSizeSmallerEvaluator
from model_evaluation.GPT2_evaluator import GPT2VLCommonsenseShapeEvaluator, GPT2VLCommonsenseColorEvaluator, \
    GPT2VLCommonsenseMaterialEvaluator, GPT2VLCommonsenseSizeLargerEvaluator, GPT2VLCommonsenseSizeSmallerEvaluator, \
    GPT2VLCommonsenseWikiMaterialEvaluator, GPT2VLCommonsenseWikiColorEvaluator, GPT2VLCommonsenseWikiShapeEvaluator, \
    GPT2EvaluatorOnHeightCommonsense, GPT2EvaluatorOnSizeCommonsense, GPT2EvaluatorOnPosrelCommonsense

if __name__ == "__main__":
    # GPT2 model evaluations
    # spatial_commonsense benchmark
    model_evaluator = GPT2EvaluatorOnHeightCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on height commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    model_evaluator = GPT2EvaluatorOnSizeCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on size commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    model_evaluator = GPT2EvaluatorOnPosrelCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on posrel commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense benchmark
    model_evaluator = GPT2VLCommonsenseShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense color benchmark
    model_evaluator = GPT2VLCommonsenseColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense material benchmark
    model_evaluator = GPT2VLCommonsenseMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki shape benchmark
    model_evaluator = GPT2VLCommonsenseWikiShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense wiki shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki color benchmark
    model_evaluator = GPT2VLCommonsenseWikiColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense wiki color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki material benchmark
    model_evaluator = GPT2VLCommonsenseWikiMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense wiki material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size larger benchmark
    model_evaluator = GPT2VLCommonsenseSizeLargerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense size larger:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size smaller benchmark
    model_evaluator = GPT2VLCommonsenseSizeSmallerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense size smaller:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # Flamingo model evaluations
    # spatial_commonsense benchmark
    model_evaluator = FlamingoEvaluatorOnHeightCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on height commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    model_evaluator = FlamingoEvaluatorOnSizeCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on size commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    model_evaluator = FlamingoEvaluatorOnPosrelCommonsense()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on posrel commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense benchmark
    # VL-commonsense color benchmark
    model_evaluator = FlamingoVLCommonsenseColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense shape benchmark
    model_evaluator = FlamingoVLCommonsenseShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense material benchmark
    model_evaluator = FlamingoVLCommonsenseMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki shape benchmark
    model_evaluator = FlamingoVLCommonsenseWikiShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense wiki shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki color benchmark
    model_evaluator = FlamingoVLCommonsenseWikiColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense wiki color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki material benchmark
    model_evaluator = FlamingoVLCommonsenseWikiMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense wiki material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size larger benchmark
    model_evaluator = FlamingoVLCommonsenseSizeLargerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense size larger:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size smaller benchmark
    model_evaluator = FlamingoVLCommonsenseSizeSmallerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"Flamingo::Accuracy on VL-commonsense size smaller:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # # GIT model evaluations
    # # spatial_commonsense benchmark
    # model_evaluator = GITEvaluatorOnHeightCommonsense()
    # evaluation_results = model_evaluator.evaluate()
    # print(f"GIT::Accuracy on height commonsense:")
    # print(evaluation_results)
    #
    # del model_evaluator
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # GIT model evaluations
    # spatial_commonsense benchmark

    # CLIP model evaluations
    # spatial_commonsense size benchmark
    model_evaluator = CLIPSpatialCommonsenseSizeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on size commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # spatial_commonsense height benchmark
    model_evaluator = CLIPSpatialCommonsenseHeightEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on height commonsense:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense shape benchmark
    model_evaluator = CLIPVLCommonsenseShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense color benchmark
    model_evaluator = CLIPVLCommonsenseColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense material benchmark
    model_evaluator = CLIPVLCommonsenseMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki shape benchmark
    model_evaluator = CLIPVLCommonsenseWikiShapeEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki shape:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki color benchmark
    model_evaluator = CLIPVLCommonsenseWikiColorEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki color:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense wiki material benchmark
    model_evaluator = CLIPVLCommonsenseWikiMaterialEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki material:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size smaller benchmark
    model_evaluator = CLIPVLCommonsenseSizeSmallerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense size smaller:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()

    # VL-commonsense size larger benchmark
    model_evaluator = CLIPVLCommonsenseSizeLargerEvaluator()
    evaluation_results = model_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense size larger:")
    print(evaluation_results)

    del model_evaluator
    gc.collect()
    torch.cuda.empty_cache()
