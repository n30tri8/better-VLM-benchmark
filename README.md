# Project Overview

This project is focused on benchmarking different large language models using two types of benchmarking datasets. The goal is to evaluate the visually grounded performance of these models.

## Project Structure

The project is organized into the following main components:

- **benchmarks/**: This folder contains classes responsible for loading and serving the benchmark datasets. These classes facilitate the evaluation process by providing the necessary data for model testing.

- **classification_logs/**: This directory stores logs generated during the model evaluation process. These logs are crucial for cases where models output ambiguous answers that cannot be automatically classified as correct or incorrect. By reviewing these logs, we can manually assess and assist in the evaluation process.

- **model_evaluation/**: This package includes individual files for each model that is being benchmarked. Each file contains the specific evaluation logic and metrics used to assess the performance of a particular model.

- **spacial_commonsense/** and **vl_commonsense/**: These folders contain the two types of benchmark datasets used in the project. The datasets are designed to test the models' understanding of viaual common sense knowledge and visual-linguistic common sense reasoning, respectively.

## Features

- **Comprehensive Benchmarking**: The project supports benchmarking of multiple large language models, providing a robust framework for performance comparison.

- **Detailed Logging**: Logs are maintained for all model outputs, especially focusing on ambiguous cases, allowing for detailed post-evaluation analysis.

- **Modular Evaluation**: Each model has a dedicated evaluation file, making it easy to add new models or modify existing evaluation criteria.

- **Diverse Datasets**: The use of two distinct benchmarking datasets ensures a comprehensive assessment of the models' capabilities in different domains of common sense reasoning.


**implementation decisions:**
- i used a simple white image for prompting flamingo model as a dummy input since it probably requires an image input
- a challenge is deciding how to prompt the models to get the best results
    - try it this link and see the difficulty: https://huggingface.co/openai-community/gpt2?text=the+typical+color+of+rice+is
- used _single_ db benchmark for better top-1 accuracy


## TODO List

- [ ] Implement evaluation based on model size
- [ ] add evaluation on wiki-{color, shep, material}
- [ ] Debug and investigate low accuracy issues
  - Compare current results with accuracies reported in the original pape
  - Identify potential discrepancies in implementation
- [ ] Complete benchmark runs and generate final report
