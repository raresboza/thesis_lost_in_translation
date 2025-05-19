# Evaluation Scripts
This directory contains the scripts to evaluate the recommender lists and compute statistical significance of the results.
It also includes plot generation and reranking strategy.

To make use of these scripts, some parts of the dataset are neeeded. The item catalogue contains pairs of `<id, language_id>` which are unique given the language identifier.
Recommendation lists contains tuples of `<user, item, predicted relevance>`. The order of the items is descending with respect to the relevance to the user.
Lastly, training data, or user history is presented as `<user, item, binary_rating>` where binary rating is 0 if original rating is less than 3.

# Environment Setup
To install the dependencies needed for running this code, execute the following command:
```
python3 -m venv .venv
```
Then, activate using:
```
source .venv/bin/activate
pip install -r requirements.txt
```
# Notes
Some of the metrics are computed by ELLIOT instead of this evaluation pipeline.
To compute the statistical significant of these results and include them in the plotting, provide the `accuracy` output dir from ELLIOT and move the REO values to the `fairness` dir.

Absolute paths were used for loading/saving the results in some instances, as the code was run on different devices and different operating systems. Please change them according to your own file structure.