# BookData-Tools Files

This directory contains instructions to carry out the language inference process used in this thesis. We have built upon the open-source project found [here](https://github.com/PIReTship/bookdata-tools).

As the project is constantly updating and the file structure is quite complex, refer to the [fork](https://github.com/raresboza/bookdata-tools) of the repository of BookData-Tools in order to benefit from our code additions and use the same version of the project. To build the right environment, follow the instructions in the README.

# How to run
After the default pipeline finishes, rerun the following commands to extract language information for the linked works. 

```
cargo run --release -- filter-marc --tag=377 --subfield=a --trim --lower -n language -o author-language.parquet viaf.parquet

cargo run --release -- cluster extract-author-language -o book-links/cluster-languages.parquet -A book-links/cluster-first-authors.parquet

cargo run --release -- cluster extract-cluster-ol-languages -o book-links/cluster-ol-work-languages.parquet
```
