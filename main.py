from model_evaluation.GPT2_evaluator import GPT2EvaluatorOnHeightCommonsense, GPT2EvaluatorOnSizeCommonsense, \
    GPT2EvaluatorOnPosrelCommonsense

if __name__ == "__main__":
    gpt2_evaluator_height_commonsense = GPT2EvaluatorOnHeightCommonsense()
    results_height_gpt2 = gpt2_evaluator_height_commonsense.evaluate()

    gpt2_evaluator_size_commonsense = GPT2EvaluatorOnSizeCommonsense()
    results_size_gpt2 = gpt2_evaluator_size_commonsense.evaluate()

    gpt2_evaluator_posrel_commonsense = GPT2EvaluatorOnPosrelCommonsense()
    results_posrel_gpt2 = gpt2_evaluator_posrel_commonsense.evaluate()
