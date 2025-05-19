# Notebooks
This directory contains the notebooks used in some of our data analysis and preparation of the dataset for the ELLIOT pipeline. They are used as follows.
- `inference.ipynb`: creates an overview of the catalogue with respect to their language characteristics and translation availability
- `user_profiles.ipynb`: computes similarity of the user activity to the book catalogue given the language characteristics and transltion availability found for the items
- `dataset.ipynb`: samples out of the primary dataset, creates comparison to the main dataset.
- `retrieve_books.ipynb`: retrieves items recommended to a user for case by case observation

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
Absolute paths were used for loading/saving the results in some instances, as the code was run on different devices and different operating systems. Please change them according to your own file structure.