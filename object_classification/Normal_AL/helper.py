from torch.utils.data import Subset
import torch
from collections import Counter

def balanced_random_ratio_split(dataset, samples_per_classes, seed):
    """
    Randomly split a balanced part from a dataset.
    Optionally fix the seed for reproducible results, e.g.:

    balanced_random_split(caltech_dataset, 0.7, seed=0)

    Arguments:
        dataset (Dataset): Dataset to be split
        samples_per_class (integer): Samples per class in the split dataset
        seed (integer): Random seed
    """
    generator = torch.Generator().manual_seed(seed)

    print("dataset.y shape: ", len(dataset.y))
    #print("dataset.y: ", dataset.y)
    # Check whether split is possible
    class_counter = Counter(dataset.y)
    if 1.0 < samples_per_classes:
        raise ValueError(f"Unable to select {samples_per_classes} ratio of images!")

    indices_balanced = []
    indices_rem = []

    starting_idx = 0
    for class_id in class_counter.keys():
        num_of_samples_in_class = class_counter[class_id]
        in_class_indices = (starting_idx + torch.randperm(num_of_samples_in_class, generator=generator)).tolist()
        starting_idx += num_of_samples_in_class

        count = int(samples_per_classes * num_of_samples_in_class)
        offset = 0
        indices_balanced.extend(in_class_indices[offset:offset + count])
        offset += count

        indices_rem.extend(in_class_indices[offset:])

    return [Subset(dataset, indices_balanced), Subset(dataset, indices_rem)]
