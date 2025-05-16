# Lost in Translation: An Investigation of Provider Fairness in Book Recommender Systems

This repository contains the code and instuctions necessary for conducting the research in the Master Thesis for Computer Science that we carried out at the TU Delft. 
We provide explanations that facilitate the reproduction of the highlighted results.

# Abstract

Recommender Systems are key instruments that are constantly being employed in the online environment
as a method of connecting users and items. Due to the resulting personalised suggestions, users benefit
from quickened decision making, however, such systems can also introduce unwanted side-effects, by
reinforcing existing biases or limiting exposure to diverse content. While much of the research surrounding
unfairness and its effects has been conducted specifically to benefit the users, the creators of the distributed
items, namely the providers, can also be subject to inequity. From the perspective of a provider, garnering
visibility and exposure of their items through these systems converts into revenue. Recommender Systems
can be a positive means in sectors where, historically, groups of providers have been disadvantaged from
reaching their desired consumers, although if mishandled can become an additional hindrance. Amongst
the many relevant domains, the book industry is a clear example where authors have been discriminated
based on traits unrelated to the quality of their writing. Publishers have manifested an adversity towards
translated works leading to an innate disadvantage when it comes to their distribution and marketing. Still,
the fairness of how recommender systems handle translated works when competing with other books
remains unexplored. To address this gap, we conduct an empirical exploration of several state of the art
recommendation algorithms, evaluating their performance with respects to accuracy and provider fairness.
This allows us to discern if any of the algorithms are a helpful conduit or a harmful one. Upon identifying
inequity concerning foreign authors, we probe the ability of mitigating it on an algorithmic level, by applying
a reranking method.

# Citation
TO-BE-FILLED-IN UPON SUBMISSION/Graduation

# Overview

This code repository is organised into 4 main directories, split by the stage of the experiment. As a result, we allow for a better understanding of the process but also to satisfy modularity of the actions themselves.
Such a separation also allows for the substitution with steps in our study with other potential practices. For instance, this could mean introducing a different dataset but carrying out the same recommendation generation and subsequent analysis, or the reverse.

We offer a brief description on the purpose of each module:
- `bookdata-tools_files`: hosts the files that were added or changed in the [BookData-Tools](https://github.com/PIReTship/bookdata-tools) project. Tasked with carrying out the inference process.
- `notebooks`: notebooks that conduct an analysis of the inferred dataset. Also contains plot generation for this section of the paper as well as the sampling and splitting of the data into train-validation-test sets.
- `elliot_files`: hosts the files that were added or changed in the [ELLIOT](https://github.com/sisinflab/elliot) project. Used for generating recommendations and part of the metric results.
- `evaluation_scripts`: contains fairness matrics, statistical significance, CP reranking implementation, plot generation for the paper


Due to this project making use of two other open-source tools to achieve its research goals, we highlight the changes that were made to them in accordance with their respective copyright policy.

# Notes

We strongly advise that any person interested in using the inference process of languages and translation availability in our study to be mindful of its limitations.
For that, please refer to the manuscript of this work where the threats to the validity of this process are discussed.

Additional documentation is present in each of the directories of this repository.