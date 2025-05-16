import numpy as np
import os

def compute_exposure_frequency(users, recommender_lists, all_items, verbose=False):
    """
    Compute exposure frequency for each item.

    Parameters:
    - users: List of users.
    - recommender_lists: {user: list of item} sorted by predicted_rating desc.
    - all_items: Set of all items being recommended.

    Returns:
    - exposure_freq: Dictionary {item: exposure_frequency}
    """
    item_exposure = {item: 0 for item in all_items}
    num_users = len(users)
    if verbose:
        print("Started computing exposure frequencies...")
    for user in users:
        ranked_items = [item for item in recommender_lists[user]]
        log_positions = 1 / np.log2(np.arange(2, len(ranked_items) + 2))

        for pos, item in enumerate(ranked_items):
            item_exposure[item] += log_positions[pos]

    for item in item_exposure:
        item_exposure[item] /= num_users

    if verbose:
        print("Finished computing exposure frequencies...")

    return item_exposure


def aggregate_item_exposures_to_groups(item_exposure, item_languages):
    """
    Aggregate item exposures to group exposures.

    Parameters:
    - item_exposure: Dictionary {item: exposure_frequency}
    - item_languages: Dictionary {item: group}

    Returns:
    - group_exposure: Dictionary {group: total_exposure}
    """
    group_exposure = {}

    for item, exposure in item_exposure.items():
        group = item_languages[item]
        if group not in group_exposure:
            group_exposure[group] = 0
        group_exposure[group] += exposure

    return group_exposure

def gini_coefficient(exposures, verbose=False, algorithm=None, cutoff=50):
    """
    Compute the Gini coefficient for exposure fairness.

    Parameters:
    - exposures: Dictionary {item: exposure_frequency}

    Returns:
    - Gini coefficient (value between 0 and 1, where 0 means perfect fairness).
    """
    if verbose:
        print("Started gini coefficient...")
    exposures = np.array(list(exposures.values()))
    exposures = np.sort(exposures)
    n = len(exposures)
    if n == 0:
        return 0

    cumulative_exposures = np.cumsum(exposures)
    sum_exposures = cumulative_exposures[-1]

    if sum_exposures == 0:
        return 0

    gini_sum = np.sum((2 * np.arange(1, n + 1) - n - 1) * exposures)
    gini = gini_sum / (n * sum_exposures)
    normalized_gini = gini / ((n - 1) / n)
    if verbose:
        print(f"Item Gini coefficient: {gini}")
        print(f"Normalized Gini: {normalized_gini}")
    if algorithm is not None:
        os.makedirs("fairness/gini", exist_ok=True)
        with open(f"fairness/gini/{algorithm}_{cutoff}.csv", 'a') as f:
            f.write(f"gini_val, {gini}\n")
            f.write(f"normalized_gini_val, {normalized_gini}\n")

    return normalized_gini, gini
