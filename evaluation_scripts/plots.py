import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

def make_activity_plot():
    all_df = pd.read_parquet(
    '/run/media/rares17/f15020b9-f291-4fe9-85f0-e6bd87520125/thesis/bookdata-tools/goodreads/gr-work-ratings.parquet',
    engine='pyarrow')
    sample_df = pd.read_parquet(
        "/home/rares17/Documents/thesis/thesis_lost_in_translation/notebooks/ratings_20000sampled_20users_10items_binary.parquet")

    user_activity1 = all_df['user'].value_counts().rename('activity').reset_index()
    user_activity1['dataset'] = 'gr_lang'

    user_activity2 = sample_df['user'].value_counts().rename('activity').reset_index()
    user_activity2['dataset'] = 'gr_lang_sample'

    # Combine for comparison
    combined_activity = pd.concat([user_activity1, user_activity2])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=combined_activity,
        x='activity',
        hue='dataset',
        fill=True,
        alpha=0.5,
        palette=['#1f77b4', '#ff7f0e'],
        common_norm=False,
        log_scale=True  # Log-transform x-axis
    )
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def make_metric_plot():
    algorithms = ['Random Baseline', 'ItemKNN', 'MultiVAE']
    metrics = ['nDCG', 'MRR', 'MAP']
    values = [
        [0.0008, 0.0005, 0.0025],  # Algorithm A
        [0.1712, 0.1018, 0.3359],  # Algorithm B
        [0.1473, 0.0818, 0.2759]  # Algorithm C
    ]

    # Convert data into a DataFrame
    data = pd.DataFrame(values, columns=metrics, index=algorithms)
    data = data.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    data.rename(columns={'index': 'Algorithm'}, inplace=True)

    # Plot grouped bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Algorithm', y='Score', hue='Metric', data=data, palette='deep')

    # Formatting
    #plt.title('Comparison of Algorithms Across Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 0.4)
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

def language_ratio_difference_plot():
    labels = ['english', 'other-translated', 'other', 'ambiguous', 'unknown']  # Mapping group numbers to strings
    ratios = [0.06135, 0.0103, 0.00445, -0.10461, 0.02851]
    algorithms = ['Algorithm 1', 'Algorithm 2', 'Algorithm 3']  # Algorithm names

    # Language ratios for each algorithm
    ratios_algo1 = [0.06135, 0.0103, 0.00445, -0.10461, 0.02851]
    ratios_algo2 = [0.04754, -0.0002, -0.0011, -0.0455, -0.0001]

    # Create DataFrame using pd.concat()
    data = pd.concat([
        pd.DataFrame({'Group': labels, 'Language Ratio Difference': ratios_algo1, 'Algorithm': 'MultiVAE'}),
        pd.DataFrame({'Group': labels, 'Language Ratio Difference': ratios_algo2, 'Algorithm': 'ItemKNN'}),
    ])

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Group', y='Language Ratio Difference', hue='Algorithm', data=data,
                palette='deep')

    # Formatting
    plt.title('Language Ratio Difference by Group')
    plt.ylabel('Language Ratio Difference')
    plt.ylim(-0.25, 0.25)  # Set y-axis domain
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()

def plot_reo():
    reo_df = pd.DataFrame({'Algorithm': ['Random Baseline', 'ItemKNN', 'MultiVAE'],
                              'REO Value': [0.08390780968785737, 0.30646829128376024, 0.5012104181987146]})

   # reo_df.drop(0, axis=0, inplace=True)
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=reo_df,
        x='Algorithm',
        y='REO Value', color='#4C72B0')
    plt.title('REO Value by Algorithm')
    plt.ylabel('REO Value')
    plt.xlabel('Algorithm')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()

def disparity_exposure_plot():
    labels = ['english', 'other-translated', 'other', 'ambiguous', 'unknown']  # Mapping group numbers to strings

    # Language ratios for each algorithm
    ratios_algo1 = [-0.15187633358693622, 0.07715294343462314, -0.0028382305428177428, 0.18065308448550246, -0.10309146379037148]
    ratios_algo2 = [-0.1517360469738267, 0.07758087160067678,  -0.0028462861522061626, 0.18020461839342936, -0.10320315686807317]

    # Create DataFrame using pd.concat()
    data = pd.concat([
        pd.DataFrame({'Group': labels, 'Disparity Exposure': ratios_algo1, 'Algorithm': 'ItemKNN'}),
        pd.DataFrame({'Group': labels, 'Disparity Exposure': ratios_algo2, 'Algorithm': 'ItemKNN+CP'}),
    ])

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Group', y='Disparity Exposure', hue='Algorithm', data=data,
                palette='deep')

    # Formatting
    plt.ylabel('Disparity Exposure')
    plt.ylim(-0.25, 0.25)  # Set y-axis domain
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()
