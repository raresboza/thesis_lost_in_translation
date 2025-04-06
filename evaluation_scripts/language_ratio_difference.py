import numpy as np

def language_ratio_difference(users, recommender_lists, training_ratings, group_tag, item_languages, verbose=False):
    """
    Compute the language ratio difference metric.

    Parameters:scores
    - users: List of users.
    - recommender_list: {user: list of (items)} sorted by predicted_rating desc
    - training_ratings: {user: list of (items)} used for training
    - group_tag: Integer ID of group.
    - item_languages: Item-based representation of a group.

    Returns:
    - Language ratio difference value.
    """
    user_ratio_difference = np.zeros(len(users))
    user_index = 0
    for user in users:
        ranked_items = [item for item in recommender_lists[user]]
        rated_items = [item for item in training_ratings[user]]

        language_ratio_training = np.sum(np.where(
            [item_languages[item] == group_tag for item in rated_items],
            1,
            -1
        )) / len (rated_items)
        language_ratio_list = np.sum(np.where(
            [item_languages[item] == group_tag for item in ranked_items],
            1,
            -1
        )) / len (ranked_items)
        user_ratio_difference[user_index] = language_ratio_training - language_ratio_list
        user_index += 1

    language_ratio_difference = np.mean(user_ratio_difference)
    if verbose:
        print(f"Language Ratio  for group {group_tag} is {language_ratio_difference}")
    return language_ratio_difference