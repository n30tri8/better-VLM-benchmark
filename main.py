from model_evaluation.CLIP_evaluator import CLIPVLCommonsenseShapeEvaluator, CLIPVLCommonsenseColorEvaluator, \
    CLIPVLCommonsenseMaterialEvaluator, CLIPVLCommonsenseWikiShapeEvaluator, CLIPVLCommonsenseWikiColorEvaluator, \
    CLIPVLCommonsenseWikiMaterialEvaluator, CLIPVLCommonsenseSizeSmallerEvaluator, CLIPVLCommonsenseSizeLargerEvaluator
from model_evaluation.Flamingo_evaluator import FlamingoEvaluatorOnHeightCommonsense, \
    FlamingoEvaluatorOnSizeCommonsense, FlamingoVLCommonsenseColorEvaluator, FlamingoVLCommonsenseMaterialEvaluator, \
    FlamingoVLCommonsenseWikiShapeEvaluator, FlamingoVLCommonsenseWikiColorEvaluator, \
    FlamingoVLCommonsenseWikiMaterialEvaluator
from model_evaluation.GPT2_evaluator import GPT2VLCommonsenseShapeEvaluator, GPT2VLCommonsenseColorEvaluator, \
    GPT2VLCommonsenseMaterialEvaluator, GPT2VLCommonsenseSizeLargerEvaluator, GPT2VLCommonsenseSizeSmallerEvaluator, \
    GPT2VLCommonsenseWikiMaterialEvaluator, GPT2VLCommonsenseWikiColorEvaluator, GPT2VLCommonsenseWikiShapeEvaluator, \
    GPT2EvaluatorOnHeightCommonsense, GPT2EvaluatorOnSizeCommonsense, GPT2EvaluatorOnPosrelCommonsense

if __name__ == "__main__":
    # GPT2 model evaluations
    # spatial_commonsense benchmark
    # gpt2_evaluator_height_commonsense = GPT2EvaluatorOnHeightCommonsense()
    # results_height_gpt2 = gpt2_evaluator_height_commonsense.evaluate()
    # print(f"GPT2::Accuracy on height commonsense:")
    # print(results_height_gpt2)
    #
    # gpt2_evaluator_size_commonsense = GPT2EvaluatorOnSizeCommonsense()
    # results_size_gpt2 = gpt2_evaluator_size_commonsense.evaluate()
    # print(f"GPT2::Accuracy on size commonsense:")
    # print(results_size_gpt2)
    #
    # gpt2_evaluator_posrel_commonsense = GPT2EvaluatorOnPosrelCommonsense()
    # results_posrel_gpt2 = gpt2_evaluator_posrel_commonsense.evaluate()
    # print(f"GPT2::Accuracy on posrel commonsense:")
    # print(results_posrel_gpt2)
    #
    # VL-commonsense benchmark
    # gpt2_vlcommonsense_shape_evaluator = GPT2VLCommonsenseShapeEvaluator()
    # results_shape_gpt2 = gpt2_vlcommonsense_shape_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense shape:")
    # print(results_shape_gpt2)
    #
    # Add VL-commonsense color benchmark
    # gpt2_vlcommonsense_color_evaluator = GPT2VLCommonsenseColorEvaluator()
    # results_color_gpt2 = gpt2_vlcommonsense_color_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense color:")
    # print(results_color_gpt2)
    #
    # # Add VL-commonsense material benchmark
    # gpt2_vlcommonsense_material_evaluator = GPT2VLCommonsenseMaterialEvaluator()
    # results_material_gpt2 = gpt2_vlcommonsense_material_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense material:")
    # print(results_material_gpt2)
    #
    # # Add VL-commonsense wiki shape benchmark
    # gpt2_vlcommonsense_wiki_shape_evaluator = GPT2VLCommonsenseWikiShapeEvaluator()
    # results_wiki_shape_gpt2 = gpt2_vlcommonsense_wiki_shape_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense wiki shape:")
    # print(results_wiki_shape_gpt2)
    #
    # # Add VL-commonsense color benchmark
    # gpt2_vlcommonsense_wiki_color_evaluator = GPT2VLCommonsenseWikiColorEvaluator()
    # results_wiki_color_gpt2 = gpt2_vlcommonsense_wiki_color_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense wiki color:")
    # print(results_wiki_color_gpt2)
    #
    # # Add VL-commonsense material benchmark
    # gpt2_vlcommonsense_wiki_material_evaluator = GPT2VLCommonsenseWikiMaterialEvaluator()
    # results_wiki_material_gpt2 = gpt2_vlcommonsense_wiki_material_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense wiki material:")
    # print(results_wiki_material_gpt2)
    #
    # Add VL-commonsense size larger benchmark
    # gpt2_vlcommonsense_size_larger_evaluator = GPT2VLCommonsenseSizeLargerEvaluator()
    # results_size_larger_gpt2 = gpt2_vlcommonsense_size_larger_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense size larger:")
    # print(results_size_larger_gpt2)
    #
    # # Add VL-commonsense size smaller benchmark
    # gpt2_vlcommonsense_size_smaller_evaluator = GPT2VLCommonsenseSizeSmallerEvaluator()
    # results_size_smaller_gpt2 = gpt2_vlcommonsense_size_smaller_evaluator.evaluate()
    # print(f"GPT2::Accuracy on VL-commonsense size smaller:")
    # print(results_size_smaller_gpt2)

    # Flamingo model evaluations
    # spatial_commonsense benchmark
    # flamingo_evaluator_height_commonsense = FlamingoEvaluatorOnHeightCommonsense()
    # results_height_flamingo = flamingo_evaluator_height_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on height commonsense:")
    # print(results_height_flamingo)

    # flamingo_evaluator_size_commonsense = FlamingoEvaluatorOnSizeCommonsense()
    # results_size_flamingo = flamingo_evaluator_size_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on size commonsense:")
    # print(results_size_flamingo)

    # flamingo_evaluator_posrel_commonsense = FlamingoEvaluatorOnPosrelCommonsense()
    # results_posrel_flamingo = flamingo_evaluator_posrel_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on posrel commonsense:")
    # print(results_posrel_flamingo)

    # VL-commonsense benchmark
    # flamingo_vlcommonsense_color_evaluator = FlamingoVLCommonsenseColorEvaluator()
    # results_color_flamingo = flamingo_vlcommonsense_color_evaluator.evaluate()
    # print(f"Flamingo::Accuracy on VL-commonsense color:")
    # print(results_color_flamingo)
    #
    # # Add VL-commonsense material benchmark
    # flamingo_vlcommonsense_material_evaluator = FlamingoVLCommonsenseMaterialEvaluator()
    # results_material_flamingo = flamingo_vlcommonsense_material_evaluator.evaluate()
    # print(f"Flamingo::Accuracy on VL-commonsense material:")
    # print(results_material_flamingo)
    #
    # # Add VL-commonsense wiki shape benchmark
    # flamingo_vlcommonsense_wiki_shape_evaluator = FlamingoVLCommonsenseWikiShapeEvaluator()
    # results_wiki_shape_flamingo = flamingo_vlcommonsense_wiki_shape_evaluator.evaluate()
    # print(f"Flamingo::Accuracy on VL-commonsense wiki shape:")
    # print(results_wiki_shape_flamingo)
    #
    # # Add VL-commonsense color benchmark
    # flamingo_vlcommonsense_wiki_color_evaluator = FlamingoVLCommonsenseWikiColorEvaluator()
    # results_wiki_color_flamingo = flamingo_vlcommonsense_wiki_color_evaluator.evaluate()
    # print(f"Flamingo::Accuracy on VL-commonsense wiki color:")
    # print(results_wiki_color_flamingo)
    #
    # # Add VL-commonsense material benchmark
    # flamingo_vlcommonsense_wiki_material_evaluator = FlamingoVLCommonsenseWikiMaterialEvaluator()
    # results_wiki_material_flamingo = flamingo_vlcommonsense_wiki_material_evaluator.evaluate()
    # print(f"Flamingo::Accuracy on VL-commonsense wiki material:")
    # print(results_wiki_material_flamingo)
    #
    # # TODO Add VL-commonsense size larger\smaller benchmark

    # GIT model evaluations
    # spatial_commonsense benchmark
    # git_evaluator_height_commonsense = GITEvaluatorOnHeightCommonsense()
    # results_height_git = git_evaluator_height_commonsense.evaluate()
    # print(f"GIT::Accuracy on height commonsense:")
    # print(results_height_git)

    # GIT model evaluations
    # spatial_commonsense benchmark

    # CLIP model evaluations
    # VL-commonsense shape benchmark
    clip_vlcommonsense_shape_evaluator = CLIPVLCommonsenseShapeEvaluator()
    results_shape_clip = clip_vlcommonsense_shape_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense shape:")
    print(results_shape_clip)

    # VL-commonsense color benchmark
    clip_vlcommonsense_color_evaluator = CLIPVLCommonsenseColorEvaluator()
    results_color_clip = clip_vlcommonsense_color_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense color:")
    print(results_color_clip)

    # VL-commonsense material benchmark
    clip_vlcommonsense_material_evaluator = CLIPVLCommonsenseMaterialEvaluator()
    results_material_clip = clip_vlcommonsense_material_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense material:")
    print(results_material_clip)

    # VL-commonsense wiki shape benchmark
    clip_vlcommonsense_wiki_shape_evaluator = CLIPVLCommonsenseWikiShapeEvaluator()
    results_wiki_shape_clip = clip_vlcommonsense_wiki_shape_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki shape:")
    print(results_wiki_shape_clip)

    # VL-commonsense wiki color benchmark
    clip_vlcommonsense_wiki_color_evaluator = CLIPVLCommonsenseWikiColorEvaluator()
    results_wiki_color_clip = clip_vlcommonsense_wiki_color_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki color:")
    print(results_wiki_color_clip)

    # VL-commonsense wiki material benchmark
    clip_vlcommonsense_wiki_material_evaluator = CLIPVLCommonsenseWikiMaterialEvaluator()
    results_wiki_material_clip = clip_vlcommonsense_wiki_material_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense wiki material:")
    print(results_wiki_material_clip)

    # VL-commonsense size smaller benchmark
    clip_vlcommonsense_size_smaller_evaluator = CLIPVLCommonsenseSizeSmallerEvaluator()
    results_size_smaller_clip = clip_vlcommonsense_size_smaller_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense size smaller:")
    print(results_size_smaller_clip)

    # VL-commonsense size larger benchmark
    clip_vlcommonsense_size_larger_evaluator = CLIPVLCommonsenseSizeLargerEvaluator()
    results_size_larger_clip = clip_vlcommonsense_size_larger_evaluator.evaluate()
    print(f"CLIP::Accuracy on VL-commonsense size larger:")
    print(results_size_larger_clip)




