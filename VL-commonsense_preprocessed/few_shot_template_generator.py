import argparse
import json
import random


def create_output_files(dataset_type, target_distribution, input_train_dataset, prompt_output_file, selected_trainset_file):
    # Read the train dataset file for processing
    with open(input_train_dataset, 'r') as infile:
        data = [json.loads(line) for line in infile]

    target_labels = target_distribution.split(',')
    grouped_data = {label: [] for label in target_labels}
    for entry in data:
        if entry['obj'] in target_labels:
            grouped_data[entry['obj']].append(entry)

    selected_entries = []
    # Randomly select 1-3 samples for each color
    for label, entries in grouped_data.items():
        if len(entries) >= 3:
            selected = random.sample(entries, 3)
            selected_entries.extend(selected)

    # Shuffle the selected samples to make the order random
    random.shuffle(selected_entries)

    with open(prompt_output_file, 'w') as outfile:
        for entry in selected_entries:
            sentence = f"The {dataset_type} of {entry['sub']} is {entry['obj']}."
            outfile.write(sentence + '\n')

        outfile.write(f"\nThe {dataset_type} of {{subject}} is")

    with open(selected_trainset_file, 'w') as selected_file:
        for entry in selected_entries:
            selected_file.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate few-shot templates.')
    parser.add_argument('--type', type=str, required=True, help='Type of the template (e.g., color, material, shape)')
    parser.add_argument('--input_file', type=str, required=True, help='Input file in JSONL format')
    parser.add_argument('--target_distribution', type=str, required=True, help='Comma-separated list of labels')
    parser.add_argument('--wiki', action='store_true', help='If to include wiki in output file names')
    args = parser.parse_args()

    input_file = args.input_file
    dataset_type = args.type
    output_file = f"{'wiki-' if args.wiki else ''}{dataset_type}-prompt-template.txt"
    selected_trainset_file = f"{'wiki-' if args.wiki else ''}{dataset_type}-selected-trainset.jsonl"

    create_output_files(dataset_type, args.target_distribution, input_file, output_file, selected_trainset_file)
