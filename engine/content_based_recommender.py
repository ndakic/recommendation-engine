import numpy as np

from sklearn.cluster import KMeans
from util import data_util


class SpotifySongRecommender:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def song_differentiation(data):
        # Songs of different genres may have similar characteristics which may affect the recommendation system.
        # This method creates a new feature that will differentiate songs from different categories.
        k_means = KMeans(n_clusters=10)
        features = k_means.fit_predict(data.select_dtypes(include=data_util.DATA_TYPES))
        data['features'] = features

    def recommend(self, liked_song, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == liked_song[0].lower()) &
                            (self.dataset.artists.str.lower() == liked_song[1].lower())].head(1).values[0]
        print("Source Song: ", song)
        rec = self.dataset[(self.dataset.name.str.lower() != liked_song[0].lower()) &
                           (self.dataset.artists.str.lower() != liked_song[1].lower())]
        for songs in rec.values:
            d = 0
            for col in np.arange(len(rec.columns)):
                if col not in [0, 1]:  # ignore name and artists columns
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]

    def print_recommendations(self, songs, total_number_of_recommendations):
        for song in songs:
            recommended_songs = self.recommend(song, total_number_of_recommendations)
            print("\nContent Based Recommendations: \n", recommended_songs)
