# nba-game-outcome-prediction

## Installation

### Install the packages
Packages can be simply installed through [pip](https://pypi.org/project/pip/):

```
pip install -r requirements.txt
```

If you prefer another file manager, like Conda, you can find the instructions to install them individually in the following links:

- [Numpy](https://numpy.org/install/)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)

### Download the data

Download and uncompress the data into your working directory (where the train.py file is) from [Kaggle](https://www.kaggle.com/datasets/nathanlauga/nba-games). All `.csv` must hang directly from that directory and NOT be inside a data directory

Make sure that all the files are downloaded:

- games.csv
- games_details.csv
- players.csv
- ranking.csv
- teams.csv

## Run the code

```
python train.py
```