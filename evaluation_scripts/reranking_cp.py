
import numpy as np
import pandas as pd
import math


def rerank_calibrated_popularity(initial_list, scores, item_popularities, user_profile, delta=0.99, k=None):
    """
    Re-rank the initial recommendations using the CP algorithm
    https://arxiv.org/abs/2103.06364

    Implementation taken from:
    https://github.com/rUngruh/mitigatingPopularityBiasInMRS
    Paper: https://doi.org/10.1145/3640457.3688102
    """
    reranked_list = []

    # save counts of each category
    category_counts = {'tail': 0,
                       'mid': 0,
                       'head': 0}

    score_count = 0

    if k == None:
        k = len(initial_list)

    # get the popularity of each item in the initial list
    item_popularities = [item_popularities[item_popularities['item'] == item]['popularity'].item() for item in
                         initial_list]

    # iterate and select and add the item with the highest criterion
    for i in range(k):
        criterion = marginal_relevances(score_count, scores, item_popularities, category_counts, len(reranked_list),
                                        user_profile, delta)

        selected_idx = np.array(criterion).argmax()

        score_count += scores[selected_idx]

        reranked_list.append(initial_list[selected_idx])
        category_counts[item_popularities[selected_idx]] += 1

        del initial_list[selected_idx]
        del scores[selected_idx]
        del item_popularities[selected_idx]
    return reranked_list


def marginal_relevances(score_count, item_scores, item_popularities, category_counts, list_len, user_profile, delta):
    """
    Computes the marginal relevance, the criterion for CP
    """
    relevances = np.zeros(len(item_scores))
    recommendation_counts = pd.DataFrame({'head_ratio': [category_counts['head']],
                                          'mid_ratio': [category_counts['mid']],
                                          'tail_ratio': [category_counts['tail']]})
    computed_categories = set()

    for i, (score, popularity) in enumerate(zip(item_scores, item_popularities)):
        if popularity in computed_categories:
            continue

        recommendation_counts[popularity + '_ratio'] += 1
        recommendation_ratios = recommendation_counts / (list_len + 1)

        relevances[i] = (
                    (1 - delta) * (score_count + score) - delta * jensen_shannon(recommendation_ratios, user_profile))
        recommendation_counts[popularity + '_ratio'] -= 1

        computed_categories.add(popularity)

    return relevances


def jensen_shannon(recommendation_ratios, user_profile):
    """
    Jensen Shannon divergence between the recommendation ratios and the user profile
    """
    epsilon = 1e-8  # Small non-zero value

    A = 0
    B = 0

    for c in ['head_ratio', 'mid_ratio', 'tail_ratio']:
        profile_ratio = user_profile[c].item()

        recommended_ratio = recommendation_ratios[c].item()

        if profile_ratio == 0:
            profile_ratio += epsilon

        if recommended_ratio == 0:
            recommended_ratio += epsilon

        A += profile_ratio * math.log2((2 * profile_ratio) / (profile_ratio + recommended_ratio))
        B += recommended_ratio * math.log2((2 * recommended_ratio) / (profile_ratio + recommended_ratio))

    js = (A + B) / 2
    return js

