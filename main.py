from model_evaluation.GPT2_evaluator import GPT2EvaluatorOnHeightCommonsense, GPT2EvaluatorOnSizeCommonsense, \
    GPT2EvaluatorOnPosrelCommonsense
from model_evaluation.GIT_evaluator import GITEvaluatorOnHeightCommonsense
from model_evaluation.Flamingo_evaluator import FlamingoEvaluatorOnHeightCommonsense

if __name__ == "__main__":
    # gpt2_evaluator_height_commonsense = GPT2EvaluatorOnHeightCommonsense()
    # results_height_gpt2 = gpt2_evaluator_height_commonsense.evaluate()
    # print(f"Accuracy on height commonsense: {results_height_gpt2}")

    # gpt2_evaluator_size_commonsense = GPT2EvaluatorOnSizeCommonsense()
    # results_size_gpt2 = gpt2_evaluator_size_commonsense.evaluate()
    # print(f"Accuracy on size commonsense: {results_size_gpt2}")
    #
    # gpt2_evaluator_posrel_commonsense = GPT2EvaluatorOnPosrelCommonsense()
    # results_posrel_gpt2 = gpt2_evaluator_posrel_commonsense.evaluate()
    # print(f"Accuracy on posrel commonsense: {results_posrel_gpt2}")

    # git_evaluator_height_commonsense = GITEvaluatorOnHeightCommonsense()
    # results_height_git = git_evaluator_height_commonsense.evaluate()
    # print(f"Accuracy on height commonsense: {results_height_git}")

    flamingo_evaluator_height_commonsense = FlamingoEvaluatorOnHeightCommonsense()
    results_height_flamingo = flamingo_evaluator_height_commonsense.evaluate()
    print(f"Accuracy on height commonsense: {results_height_flamingo}")
#     TODO i modified src/factory in open_flamingo. how to replicate this in another environment?


