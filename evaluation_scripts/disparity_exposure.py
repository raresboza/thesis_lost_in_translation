import numpy as np

def disparate_exposure(users, recommender_lists, group_tag, item_languages, verbose=False):
    """
    Compute the disparate exposure metric.

    Parameters:
    - users: List of users.
    - recommender_list: {user: list of (item, predicted_rating)} sorted by predicted_rating desc
    - group_tag: Integer ID of group.
    - item_languages: Item-based representation of a group.

    Returns:
    - Disparate exposure value.
    """
    user_exposures = np.zeros(len(users))
    group_representation = list(item_languages.values()).count(group_tag) / len(list(item_languages.keys()))
    if verbose:
        print(f"Group representation is {group_representation}")
    user_index = 0
    for user in users:
        ranked_items = [item for item in recommender_lists[user]]

        numerator = np.sum([1 / np.log2(ranked_items.index(i) + 2) for i in ranked_items if item_languages[i] == group_tag])
        denominator = np.sum([1 / np.log2(pos + 2) for pos in range(len(ranked_items))])

        user_exposures[user_index] = numerator / denominator if denominator > 0 else 0
        user_index += 1

    group_exposure = np.mean(user_exposures)
    disparate_exposure = group_exposure - group_representation
    upperbound = 1-group_representation
    if verbose:
        print (f"Bound for exposure is [-{group_representation}, {upperbound}] ")
        print(f"Disparate exposure for group {group_tag} is {disparate_exposure}")
    return disparate_exposure