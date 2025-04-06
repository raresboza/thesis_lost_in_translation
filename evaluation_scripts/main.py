import pandas as pd

from evaluation_scripts.plots import language_ratio_difference_plot
from plots import make_metric_plot, language_ratio_difference_plot, plot_reo, disparity_exposure_plot
from evaluation_scripts.disparity_exposure import disparate_exposure
from gini_coefficient import compute_exposure_frequency, aggregate_item_exposures_to_groups , gini_coefficient
from popularity import compute_popularity, create_user_profile_popularity
from reranking_cp import rerank_calibrated_popularity
from language_ratio_difference import language_ratio_difference
# item language mapping
# eng-original                - 0
# other-translated            - 1
# other-translation-not-found - 2
# ambiguous                   - 3
# unknown                     - 4

def main():
    # read training data
    training_df = pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None)
    training_dict = \
    pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None).groupby(0)[
        1].apply(list).to_dict()
    # read recommendation list and scores
    recs = pd.read_csv(
        '/home/rares17/Documents/thesis/thesis_lost_in_translation/elliot/data/sim_res/recs/ItemKNN_nn=8_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
        sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
    scores = pd.read_csv(
        '/home/rares17/Documents/thesis/thesis_lost_in_translation/elliot/data/sim_res/recs/ItemKNN_nn=8_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
        sep='\t', header=None).groupby(0)[2].apply(list).to_dict()
    # read item language pairs
    item_languages_df = pd.read_csv('item_languages.tsv', sep='\t')
    item_languages = dict(zip(item_languages_df['item'], item_languages_df['deduced_language']))


    # fairness evaluation
    #gini
    #disparity
    #language ratio difference
    item_exposure_frequencies = compute_exposure_frequency(recs.keys(), recs, set(item_languages.keys()), True)
    group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
    normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies)
    print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
    normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies)
    print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
    for i in range(0, 5):
        language_ratio_difference(recs.keys(), recs, training_dict, i, item_languages, True)
        disparate_exposure(set(recs.keys()), recs, i, item_languages, True)
    # reranking
    reranked_recs = rerank_lists(training_df, recs, scores)
    # post-reranking evaluation
    item_exposure_frequencies = compute_exposure_frequency(reranked_recs.keys(), reranked_recs, set(item_languages.keys()), True)
    group_exposure_frequencies = aggregate_item_exposures_to_groups(item_exposure_frequencies, item_languages)
    normalized_item_gini, item_gini = gini_coefficient(item_exposure_frequencies)
    print(f"Normalized item Gini is: {normalized_item_gini}. Non-normalized is {item_gini}.")
    normalized_group_gini, group_gini = gini_coefficient(group_exposure_frequencies)
    print(f"Normalized group Gini is: {normalized_group_gini}. Non-normalized is {group_gini}.")
    for i in range(0, 5):
        language_ratio_difference(reranked_recs.keys(), reranked_recs, training_dict, i, item_languages, True)
        disparate_exposure(set(reranked_recs.keys()), reranked_recs, i, item_languages, True)
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
                                     delta=0.99)
        processed_user_count += 1
        print(processed_user_count)
    return reranked_recs
def compute_difference_language():
    item_languages_df = pd.read_csv('item_languages.tsv', sep='\t')
    item_languages = dict(zip(item_languages_df['item'], item_languages_df['deduced_language']))
    training = pd.read_csv("ratings_20000sampled_20users_10items_split/train_dataset.tsv", sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
    #recs = pd.read_csv('/home/rares17/Documents/thesis/thesis_lost_in_translation/elliot/data/sim_res/recs/MultiVAE_seed=42_e=19_bs=128_intermediate_dim=778_latent_dim=101_reg_lambda=0$0080536686573167_lr=0$0015655504610557827_dropout_pkeep=0$5_it=19.tsv', sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
    recs = pd.read_csv(
        '/home/rares17/Documents/thesis/thesis_lost_in_translation/elliot/data/sim_res/recs/ItemKNN_nn=8_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv',
        sep='\t', header=None).groupby(0)[1].apply(list).to_dict()
    users = recs.keys()
    print(recs)

    for i in range(0, 5):
        language_ratio_difference(users, recs, training, i, item_languages, True)

if __name__ == "__main__":
    main()
    #compute_difference_language()
    #language_ratio_difference_plot()
    #make_metric_plot()
    # plot_reo()
    # disparity_exposure_plot()

