from model_evaluation.GPT2_evaluator import GPT2EvaluatorOnHeightCommonsense

if __name__ == "__main__":
    gpt2_evaluator_height_commonsense = GPT2EvaluatorOnHeightCommonsense()
    results = gpt2_evaluator_height_commonsense.evaluate()
    print(results)
