import pandas as pd
import matplotlib.pyplot as plt


def compute_popularity(ratings_df, verbose = False):
    raw_popularity_df = ratings_df.groupby(1).size().reset_index(name='num_ratings')
    if verbose:
        book_popularity_sorted = raw_popularity_df.sort_values('num_ratings', ascending=False)
        print(book_popularity_sorted)

    quantiles = raw_popularity_df['num_ratings'].quantile([0.5, 0.8]).values
    q50, q80 = quantiles[0], quantiles[1]

    def categorize_popularity(num_ratings):
        if num_ratings >= q80:
            return 'head'
        elif num_ratings >= q50:
            return 'mid'
        else:
            return 'tail'

    raw_popularity_df['popularity'] = raw_popularity_df['num_ratings'].apply(categorize_popularity)

    if verbose:
        print(raw_popularity_df['popularity'].value_counts(normalize=True))

    # prepare for rerank
    item_popularity = raw_popularity_df[[1, 'popularity']].rename(columns={1: 'item'})
    return item_popularity


def create_user_profile_popularity(rating_df, item_popularity_df, verbose = False):
    user_history_df = rating_df[[0,1]].rename(columns={0: 'user', 1: 'item'})
    user_item_popularity_df = pd.merge(user_history_df, item_popularity_df, on='item', how='left')

    user_profile_ratios_df = (
        user_item_popularity_df
        .groupby(['user', 'popularity'])
        .size()
        .unstack(fill_value=0)
        .assign(
            total=lambda x: x.sum(axis=1),
            head_ratio=lambda x: x['head'] / x['total'],
            mid_ratio=lambda x: x['mid'] / x['total'],
            tail_ratio=lambda x: x['tail'] / x['total']
        )
        [['head_ratio', 'mid_ratio', 'tail_ratio']]  # keep only ratios
        .reset_index()
    )

    user_profiles = {
        user: pd.DataFrame({
            'head_ratio': [row['head_ratio']],
            'mid_ratio': [row['mid_ratio']],
            'tail_ratio': [row['tail_ratio']]
        })
        for user, row in user_profile_ratios_df.set_index('user').iterrows()
    }

    if verbose:
        # plot for a user
        ratios = user_profile_ratios_df.set_index('user').loc[105]
        ratios[['head_ratio', 'mid_ratio', 'tail_ratio']].plot(kind='bar', figsize=(6, 4))
        plt.title(f'User {105} Profile')
        plt.ylabel('Ratio')
        plt.xticks(rotation=0)
        plt.show()

    return user_profiles
