import argparse
import random


def main(output_file, train_dataset, target_distribution):
    # Convert the comma-separated list of labels into a list
    target_distribution = target_distribution.split(',')

    # Read the train dataset file for processing
    with open(train_dataset, 'r') as file:
        data = file.readlines()

    # Filter lines that contain the target colors
    filtered_samples = {label: [] for label in target_distribution}
    for line in data:
        for color in target_distribution:
            if color in line:
                filtered_samples[color].append(line.strip())

    # Randomly select 1-3 samples for each color
    final_samples = []
    for color, samples in filtered_samples.items():
        final_samples.extend(random.sample(samples, min(3, len(samples))))

    # Shuffle the selected samples to make the order random
    random.shuffle(final_samples)

    # Join the samples into the desired prompt format
    few_shot_prompt = "\n".join(final_samples)

    # Write the few_shot_prompt to a file
    with open(output_file, 'w') as file:
        file.write(few_shot_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate few-shot prompt from dataset.')
    parser.add_argument('--output_file', type=str, required=True, help='The name of the output file')
    parser.add_argument('--train_dataset', type=str, required=True, help='The name of the train dataset file')
    parser.add_argument('--target_distribution', type=str, required=True, help='Comma-separated list of labels')

    args = parser.parse_args()
    main(args.output_file, args.train_dataset, args.target_distribution)
