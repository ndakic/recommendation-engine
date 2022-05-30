import pandas as pd

from scipy.sparse import csr_matrix
from spotify_engine.content_based_recommender import CBRecommender
from spotify_engine.collaborative_recommender import CFRecommender
from util import data_util


class HybridRecommender:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        self.cb_recommender = None
        self.cf_recommender = None

    def init_cb_recommender(self, data):
        CBRecommender.song_differentiation(data)
        data_util.data_normalization(data)
        self.cb_recommender = CBRecommender(data)

    def get_cb_recommendations(self, song):
        return self.cb_recommender.recommend(song, self.number_of_recommendations)

    def init_cf_recommender(self, tracks_dataset, liked_songs_dataset):
        merged_datasets = pd.merge(left=liked_songs_dataset, right=tracks_dataset, how="inner", on="track_id")
        df_songs_features = merged_datasets.pivot(index='track_id', columns='user_id', values='listen_count').fillna(0)
        mat_songs_features = csr_matrix(df_songs_features.values)
        df_unique_tracks = tracks_dataset.drop_duplicates(subset=['track_id']).reset_index(drop=True)[['track_id', 'name']]
        decode_id_song = {
            song: i for i, song in
            enumerate(list(df_unique_tracks.set_index('track_id').loc[df_songs_features.index].name))
        }
        self.cf_recommender = CFRecommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)

    def get_cf_recommendations(self, song_name):
        return self.cf_recommender.make_recommendation(new_song=song_name,
                                                       n_recommendations=self.number_of_recommendations)

    def recommend(self, song):
        content_based_recommendations = self.get_cb_recommendations(song)
        collaboration_filtering_recommendations = self.get_cf_recommendations(song[0])
        cb_df_object = pd.DataFrame(data=collaboration_filtering_recommendations, columns=['name'])
        result = pd.concat([content_based_recommendations, cb_df_object]).sample(frac=1).reset_index(drop=True)
        print("Hybrid Recommendations: \n")
        print(result)




