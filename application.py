import pandas as pd

from engine import content_based_recommender
from util import data_util

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

FILENAME = "data/tracks_features.csv"


def find_song(song_name):
    print(data[data['name'] == song_name])


if __name__ == "__main__":
    print("Processing started..")
    data = data_util.load_data(FILENAME)

    SONGS_THAT_USER_LIKES = [('The Power Of Love', "['Jenifer Jackson']"), ("Suspicious Minds", "['Elvis Presley']")]
    content_based_recommender.SpotifySongRecommender.song_differentiation(data)
    data_util.data_normalization(data)
    spotifySongRecommender = content_based_recommender.SpotifySongRecommender(data)
    spotifySongRecommender.print_recommendations(SONGS_THAT_USER_LIKES, 5)

    print("Processing finished..")

