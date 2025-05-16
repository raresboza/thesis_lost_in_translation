import pandas as pd
import os

from evaluation_scripts.plots import language_ratio_difference_plot
from plots import make_metric_plot, language_ratio_difference_plot, plot_reo, disparity_exposure_plot
from evaluation_scripts.disparity_exposure import disparate_exposure
from gini_coefficient import compute_exposure_frequency, aggregate_item_exposures_to_groups, gini_coefficient
from popularity import compute_popularity, create_user_profile_popularity
from reranking_cp import rerank_calibrated_popularity
from language_ratio_difference import language_ratio_difference
from significance import wilcoxon_test
from plots import plot_significance_matrices, plot_significance_matrices_with_groups, boxplot_comparison, violin_comparison
# item language mapping
# eng-original                - 0
# other-translated            - 1
# other-translation-not-found - 2
# ambiguous                   - 3
# unknown                     - 4

def main():
    # read training data
    training_dict = \
    pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None).groupby(0)[
        1].apply(list).to_dict()
    # read item language pairs
    item_languages_df = pd.read_csv('item_languages.tsv', sep='\t')
    item_languages = dict(zip(item_languages_df['item'], item_languages_df['deduced_language']))
    for recommender_algorithm in ['random', 'item_knn', 'pmf', 'bprmf', 'multivae', 'slim', 'mf2020', 'itemautorec']:
        # read recommendation list and scores
        recs = pd.read_csv(
            f'recs/{recommender_algorithm}.tsv',
            sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
        recs_top10 = cut_recommendations_to_size(recs)
        # fairness metrics top 50 original
        item_exposure_frequencies = compute_exposure_frequency(recs.keys(), recs, set(item_languages.keys()), True)
        group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
        normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies, algorithm=recommender_algorithm)
        print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
        normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies, algorithm=recommender_algorithm)
        print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
        for i in range(0, 5):
            language_ratio_difference(recs.keys(), recs, training_dict, i, item_languages, True, recommender_algorithm)
            disparate_exposure(recs.keys(), recs, i, item_languages, True, recommender_algorithm)
        # fairness metrics top 10 original
        item_exposure_frequencies = compute_exposure_frequency(recs_top10.keys(), recs_top10, set(item_languages.keys()), True)
        group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
        normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies, algorithm=recommender_algorithm, cutoff=10)
        print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
        normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies,
                                                             algorithm=recommender_algorithm,
                                                             cutoff=10)
        print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
        for i in range(0, 5):
            language_ratio_difference(recs_top10.keys(), recs_top10, training_dict, i, item_languages, True, recommender_algorithm, cutoff=10)
            disparate_exposure(recs_top10.keys(), recs_top10, i, item_languages, True, recommender_algorithm, cutoff=10)
        # reranking
        reranked_recs = pd.read_csv(
            f'recs/reranked_recs/{recommender_algorithm}.tsv',
            sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
        reranked_recs_top10 = cut_recommendations_to_size(reranked_recs)
        # fairness metrics top 50 rerank
        item_exposure_frequencies = compute_exposure_frequency(reranked_recs.keys(), reranked_recs, set(item_languages.keys()), True)
        group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
        normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies, algorithm=f"{recommender_algorithm}_reranked")
        print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
        normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies, algorithm=f"{recommender_algorithm}_reranked")
        print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
        for i in range(0, 5):
            language_ratio_difference(reranked_recs.keys(), reranked_recs, training_dict, i, item_languages, True,f"{recommender_algorithm}_reranked")
            disparate_exposure(reranked_recs.keys(), reranked_recs, i, item_languages, True, f"{recommender_algorithm}_reranked")
        # fairness metrics top 10 rerank
        item_exposure_frequencies = compute_exposure_frequency(reranked_recs_top10.keys(), reranked_recs_top10,
                                                               set(item_languages.keys()), True)
        group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
        normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies,
                                                           algorithm=f"{recommender_algorithm}_reranked",
                                                           cutoff=10)
        print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
        normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies,
                                                             algorithm=f"{recommender_algorithm}_reranked",
                                                             cutoff=10)
        print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
        for i in range(0, 5):
            language_ratio_difference(reranked_recs_top10.keys(), reranked_recs_top10, training_dict, i, item_languages, True,
                                      f"{recommender_algorithm}_reranked", cutoff=10)
            disparate_exposure(reranked_recs_top10.keys(), reranked_recs_top10, i, item_languages, True,
                               f"{recommender_algorithm}_reranked", cutoff=10)
    #====================TESTING GROUND======================
    # item_languages_df = pd.read_csv('item_languages.tsv', sep='\t')
    # item_languages = dict(zip(item_languages_df['item'], item_languages_df['deduced_language']))
    #knn_recs = pd.read_csv('item_knn_recs.tsv', sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
    #knn_scores = pd.read_csv('item_knn_recs.tsv', sep='\t', header=None).groupby(0)[2].apply(list).to_dict()
    # disparate_exposure(set(knn_recs.keys()), knn_recs, 0, item_languages, True)


    # training_df = pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None)
    # rerank_lists(training_df, knn_recs, knn_scores)
    #item_exposure_frequency = compute_exposure_frequency(set(knn_recs.keys()), knn_recs, set(item_languages.keys()))
    # print(gini_coefficient([*item_exposure_frequency.values()]))
    # df = pd.read_parquet("/home/rares17/Documents/thesis/thesis_lost_in_translation/notebooks/ratings_20000sampled_20users_10items_binary.parquet")
    # print(df[['item', 'deduced_language']])
    # df = df[['item', 'deduced_language']]
    # df = df.drop_duplicates(subset=['item'], keep='last')
    # df['deduced_language'] = df['deduced_language'].map({'eng-original': 0, 'other-translated': 1, 'other-translation-not-found': 2, 'ambiguous': 3, 'unknown': 4})
    # df.to_csv("item_languages.tsv", sep='\t', index=False)

def rerank_lists(user_history_df, recommendation_lists_dict, recommendation_scores_dict):
    reranked_recs = {}
    item_popularity_df = compute_popularity(user_history_df, True)
    user_profiles_popularity_df = create_user_profile_popularity(user_history_df, item_popularity_df, True)
    processed_user_count = 0
    for user, items in recommendation_lists_dict.items():
        reranked_recs[user] = rerank_calibrated_popularity(initial_list=recommendation_lists_dict[user],
                                     scores=recommendation_scores_dict[user],
                                     item_popularities=item_popularity_df,
                                     user_profile=user_profiles_popularity_df[user],
                                     delta=0.90)
        processed_user_count += 1
        print(processed_user_count)
    return reranked_recs

def save_reranked_lists():
    # read training data
    training_df = pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None)
    os.makedirs("recs/reranked_recs", exist_ok=True)
    for recommender_algorithm in ['random', 'item_knn', 'pmf', 'bprmf', 'multivae', 'slim', 'mf2020', 'itemautorec']:
        recs = pd.read_csv(
            f'recs/{recommender_algorithm}.tsv',
            sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
        scores = pd.read_csv(
            f'recs/{recommender_algorithm}.tsv',
            sep='\t', header=None).groupby(0)[2].apply(list).to_dict()
        if recommender_algorithm == 'item_knn':
            scores = {
                user: [(s - 0.0) / (10.0 - 0.0) for s in local_scores]
                for user, local_scores in scores.items()
            }
        elif recommender_algorithm == 'bprmf':
            scores = {
                user: [(s - 0.0) / (15.0 - 0.0) for s in local_scores]
                for user, local_scores in scores.items()
            }
        elif recommender_algorithm == 'multivae':
            scores = {
                user: [(s + 15.0) / (0.0 + 15.0) for s in local_scores]
                for user, local_scores in scores.items()
            }
        elif recommender_algorithm == 'mf2020':
            scores = {
                user: [(s - 0.0) / (5.0 - 0.0) for s in local_scores]
                for user, local_scores in scores.items()
            }
        reranked_recs = rerank_lists(training_df, recs, scores)
        data = [(user, item) for user, items in reranked_recs.items() for item in items]
        df = pd.DataFrame(data)
        df.to_csv(f"recs/reranked_recs/{recommender_algorithm}.tsv", sep='\t', header=False, index=False)

def compute_significance():
    algorithms=['random', 'item_knn', 'pmf', 'bprmf', 'multivae', 'slim', 'mf2020', 'itemautorec']
    # accuracy metrics original + reo total
    accuracy_significance = {}
    for accuracy_metric in ['ndcg', 'map', 'mrr']:
        scores = []
        for algorithm in algorithms:
                metric_values = pd.read_csv(f'evaluation/{algorithm}/{accuracy_metric}_50.tsv',sep='\t')
                scores.append(list(metric_values['value']))
        significance_matrix = wilcoxon_test(scores)
        accuracy_significance[accuracy_metric] = [significance_matrix]
    reo_total_scores = []
    for algorithm in algorithms:
        metric_values = pd.read_csv(f'fairness/reo/{algorithm}/group_5_50.tsv', sep='\t')
        reo_total_scores.append(list(metric_values['value']))
    significance_matrix = wilcoxon_test(scores)
    accuracy_significance['reo'] = [significance_matrix]
    plot_significance_matrices(accuracy_significance)
    # fairness metrics original
    groups = ['eng-original', 'other-translated', 'other-not-translated', 'ambiguous', 'unknown']
    fairness_significance = {}
    for fairness_metric in ['disp_exp', 'lrd', 'reo']:
        fairness_significance[fairness_metric] = []
        group_size = 5
        if fairness_metric == 'reo':
            group_size = 5
        for group_tag in range(0, group_size):
            scores = []
            for algorithm in algorithms:
                metric_values = pd.read_csv(f'fairness/{fairness_metric}/{algorithm}/group_{group_tag}_50.tsv',sep='\t')
                scores.append(list(metric_values['value']))
            significance_matrix = wilcoxon_test(scores)
            fairness_significance[fairness_metric].append(significance_matrix)
    plot_significance_matrices_with_groups(fairness_significance, groups)

    # check original list to reranked counterpart
    accuracy_reranking_significance = []
    for accuracy_metric in ['ndcg', 'map', 'mrr']:
        for algorithm in algorithms:
            for list_size in [10, 50]:
                original_values = pd.read_csv(f'evaluation/{algorithm}/{accuracy_metric}_{list_size}.tsv',sep='\t')
                reranked_values = pd.read_csv(f'evaluation_reranked/{algorithm}/{accuracy_metric}_{list_size}.tsv', sep='\t')
                scores = [list(original_values['value']), list(reranked_values['value'])]
                significance_matrix = wilcoxon_test(scores)
                accuracy_reranking_significance.append((accuracy_metric, algorithm, list_size, significance_matrix[0][1]))
    pd.DataFrame(accuracy_reranking_significance).to_csv(f"fairness/accuracy_significance.tsv", sep='\t', index=False)

    fairness_reranking_significance = []
    for fairness_metric in ['disp_exp', 'lrd', 'reo']:
        for list_size in [10, 50]:
            group_size = 5
            if fairness_metric == 'reo':
                group_size = 6
            for group_tag in range(0, group_size):
                for algorithm in algorithms:
                    original_values = pd.read_csv(f'fairness/{fairness_metric}/{algorithm}/group_{group_tag}_{list_size}.tsv', sep='\t')
                    reranked_values = pd.read_csv(f'fairness/{fairness_metric}/{algorithm}_reranked/group_{group_tag}_{list_size}.tsv', sep='\t')
                    scores = [list(original_values['value']), list(reranked_values['value'])]
                    significance_matrix = wilcoxon_test(scores)
                    fairness_reranking_significance.append(
                        (fairness_metric, int(group_tag), int(list_size), int(significance_matrix[0][1]), algorithm))
    pd.DataFrame(fairness_reranking_significance).to_csv(f"fairness/fairness_significance.tsv", sep='\t', index=False)

def cut_recommendations_to_size(recs, size=10):
    return {user: items[:10] for user, items in recs.items()}

def compare_algorithms():
    slim_df = pd.read_csv(f'evaluation/slim/ndcg_50.tsv',sep='\t')
    slim_df['Algorithm'] = 'Slim'
    itemknn_df = pd.read_csv(f'evaluation/item_knn/ndcg_50.tsv', sep='\t')
    itemknn_df['Algorithm'] = 'ItemKNN'
    multivae_df = pd.read_csv(f'evaluation/multivae/ndcg_50.tsv', sep='\t')
    multivae_df['Algorithm'] ='MultiVAE'
    boxplot_comparison(slim_df, itemknn_df, multivae_df, 'Algorithm', 'value','nDCG')

    de_0 = pd.read_csv(f'fairness/disp_exp/slim/group_0_50.tsv', sep='\t')
    de_0['Metric'] = 'Disparity Exposure'
    de_0['Type'] = 'Original'
    de_0['Group'] = 'eng-original'
    de_1 = pd.read_csv(f'fairness/disp_exp/slim/group_1_50.tsv', sep='\t')
    de_1['Metric'] = 'Disparity Exposure'
    de_1['Type'] = 'Original'
    de_1['Group'] = 'other-translated'
    de_3 = pd.read_csv(f'fairness/disp_exp/slim/group_3_50.tsv', sep='\t')
    de_3['Metric'] = 'Disparity Exposure'
    de_3['Type'] = 'Original'
    de_3['Group'] = 'ambiguous'
    boxplot_comparison(de_0, de_1, de_3, 'Group', 'value', 'Disparate Exposure', -1)
    lrd_0 = pd.read_csv(f'fairness/lrd/slim/group_0_50.tsv', sep='\t')
    lrd_0['Metric'] = 'Language Ratio Difference'
    lrd_0['Type'] = 'Original'
    lrd_0['Group'] = 'eng-original'
    lrd_1 = pd.read_csv(f'fairness/lrd/slim/group_1_50.tsv', sep='\t')
    lrd_1['Metric'] = 'Language Ratio Difference'
    lrd_1['Type'] = 'Original'
    lrd_1['Group'] = 'other-translated'
    lrd_3 = pd.read_csv(f'fairness/lrd/slim/group_3_50.tsv', sep='\t')
    lrd_3['Metric'] = 'Language Ratio Difference'
    lrd_3['Type'] = 'Original'
    lrd_3['Group'] = 'ambiguous'
    boxplot_comparison(lrd_0, lrd_1, lrd_3, 'Group', 'value', 'Language Ratio Difference', -1)
    #only 10
    de_0 = pd.read_csv(f'fairness/disp_exp/slim/group_0_10.tsv', sep='\t')
    de_0['Metric'] = 'Disparity Exposure'
    de_0['Type'] = 'Original'
    de_0['Group'] = 'eng-original'
    de_1 = pd.read_csv(f'fairness/disp_exp/slim/group_1_10.tsv', sep='\t')
    de_1['Metric'] = 'Disparity Exposure'
    de_1['Type'] = 'Original'
    de_1['Group'] = 'other-translated'
    de_3 = pd.read_csv(f'fairness/disp_exp/slim/group_3_10.tsv', sep='\t')
    de_3['Metric'] = 'Disparity Exposure'
    de_3['Type'] = 'Original'
    de_3['Group'] = 'ambiguous'
    lrd_0 = pd.read_csv(f'fairness/lrd/slim/group_0_10.tsv', sep='\t')
    lrd_0['Metric'] = 'Language Ratio Difference'
    lrd_0['Type'] = 'Original'
    lrd_0['Group'] = 'eng-original'
    lrd_1 = pd.read_csv(f'fairness/lrd/slim/group_1_10.tsv', sep='\t')
    lrd_1['Metric'] = 'Language Ratio Difference'
    lrd_1['Type'] = 'Original'
    lrd_1['Group'] = 'other-translated'
    lrd_3 = pd.read_csv(f'fairness/lrd/slim/group_3_10.tsv', sep='\t')
    lrd_3['Metric'] = 'Language Ratio Difference'
    lrd_3['Type'] = 'Original'
    lrd_3['Group'] = 'ambiguous'
    # reranked
    de_0_reranked = pd.read_csv(f'fairness/disp_exp/slim_reranked/group_0_10.tsv', sep='\t')
    de_0_reranked['Metric'] = 'Disparity Exposure'
    de_0_reranked['Type'] = 'Reranked'
    de_0_reranked['Group'] = 'eng-original'
    de_1_reranked = pd.read_csv(f'fairness/disp_exp/slim_reranked/group_1_10.tsv', sep='\t')
    de_1_reranked['Metric'] = 'Disparity Exposure'
    de_1_reranked['Type'] = 'Reranked'
    de_1_reranked['Group'] = 'other-translated'
    de_3_reranked = pd.read_csv(f'fairness/disp_exp/slim_reranked/group_3_10.tsv', sep='\t')
    de_3_reranked['Metric'] = 'Disparity Exposure'
    de_3_reranked['Type'] = 'Reranked'
    de_3_reranked['Group'] = 'ambiguous'
    lrd_0_reranked = pd.read_csv(f'fairness/lrd/slim_reranked/group_0_10.tsv', sep='\t')
    lrd_0_reranked['Metric'] = 'Language Ratio Difference'
    lrd_0_reranked['Type'] = 'Reranked'
    lrd_0_reranked['Group'] = 'eng-original'
    lrd_1_reranked = pd.read_csv(f'fairness/lrd/slim_reranked/group_1_10.tsv', sep='\t')
    lrd_1_reranked['Metric'] = 'Language Ratio Difference'
    lrd_1_reranked['Type'] = 'Reranked'
    lrd_1_reranked['Group'] = 'other-translated'
    lrd_3_reranked = pd.read_csv(f'fairness/lrd/slim_reranked/group_3_10.tsv', sep='\t')
    lrd_3_reranked['Metric'] = 'Language Ratio Difference'
    lrd_3_reranked['Type'] = 'Reranked'
    lrd_3_reranked['Group'] = 'ambiguous'
    violin_comparison(pd.concat([de_0, de_1, de_3, de_0_reranked, de_1_reranked, de_3_reranked]))
    violin_comparison(pd.concat([lrd_0, lrd_1, lrd_3, lrd_0_reranked, lrd_1_reranked, lrd_3_reranked]))
    # reo_0 = pd.read_csv(f'fairness/reo/slim/group_0_50.tsv', sep='\t')
    # reo_0['Metric'] = 'Ranked Equal Opportunity'
    # reo_0['Group'] = 'eng-original'
    # reo_1 = pd.read_csv(f'fairness/reo/slim/group_1_50.tsv', sep='\t')
    # reo_1['Metric'] = 'Ranked Equal Opportunity'
    # reo_1['Group'] = 'other-translated'
    # violin_comparison(pd.concat([de_0, de_1, lrd_0, lrd_1, reo_0, reo_1], ignore_index=True))

if __name__ == "__main__":
    #main()
    #save_reranked_lists()
    compare_algorithms()
    #compute_significance()


