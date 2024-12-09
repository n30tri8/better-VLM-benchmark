from model_evaluation.GPT2_evaluator import GPT2VLCommonsenseShapeEvaluator, GPT2VLCommonsenseColorEvaluator, \
    GPT2VLCommonsenseMaterialEvaluator

if __name__ == "__main__":
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

    # git_evaluator_height_commonsense = GITEvaluatorOnHeightCommonsense()
    # results_height_git = git_evaluator_height_commonsense.evaluate()
    # print(f"GIT::Accuracy on height commonsense:")
    # print(results_height_git)

    # flamingo_evaluator_height_commonsense = FlamingoEvaluatorOnHeightCommonsense()
    # results_height_flamingo = flamingo_evaluator_height_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on height commonsense:")
    # print(results_height_flamingo)
    #
    # flamingo_evaluator_size_commonsense = FlamingoEvaluatorOnSizeCommonsense()
    # results_size_flamingo = flamingo_evaluator_size_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on size commonsense:")
    # print(results_size_flamingo)
    #
    # flamingo_evaluator_posrel_commonsense = FlamingoEvaluatorOnPosrelCommonsense()
    # results_posrel_flamingo = flamingo_evaluator_posrel_commonsense.evaluate()
    # print(f"Flamingo::Accuracy on posrel commonsense:")
    # print(results_posrel_flamingo)

    gpt2_vlcommonsense_shape_evaluator = GPT2VLCommonsenseShapeEvaluator()
    results_shape_gpt2 = gpt2_vlcommonsense_shape_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense shape:")
    print(results_shape_gpt2)

    # Add VL-commonsense color benchmark
    gpt2_vlcommonsense_color_evaluator = GPT2VLCommonsenseColorEvaluator()
    results_color_gpt2 = gpt2_vlcommonsense_color_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense color:")
    print(results_color_gpt2)

    # Add VL-commonsense material benchmark
    gpt2_vlcommonsense_material_evaluator = GPT2VLCommonsenseMaterialEvaluator()
    results_material_gpt2 = gpt2_vlcommonsense_material_evaluator.evaluate()
    print(f"GPT2::Accuracy on VL-commonsense material:")
    print(results_material_gpt2)
