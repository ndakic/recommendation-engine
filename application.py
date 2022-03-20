import pandas as pd
import warnings

from engine import content_based_recommender
from util import data_util
from scipy.sparse import csr_matrix
from engine import collaborative_recommender

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)

FILENAME = "data/tracks_features.csv"


def find_song(data, song_name):
    print(data[data['name'] == song_name])


def content_based_recommendation(data):
    # There is two types of content based recommendation
    # First way:
    #     - calculate how similar are items (songs) with previously liked/rated item (song).
    #     - choose n most similar songs.
    # Second way:
    #     - make Item Profile for all songs.
    #     - make User Profile from previously liked/rated items (songs).
    #     - recommend Items (songs) based on User Profile (user profile tell us which attributes are important to user)
    # In this project, first way is used.
    mocked_song_that_user_likes = [("Suspicious Minds", "['Elvis Presley']")]
    content_based_recommender.SpotifySongRecommender.song_differentiation(data)
    data_util.data_normalization(data)
    spotify_song_recommender = content_based_recommender.SpotifySongRecommender(data)
    spotify_song_recommender.print_recommendations(mocked_song_that_user_likes, 5)


def data_exploration(data):
    # print top 10 mostly liked songs
    top_listening_songs = data.groupby(['name', 'artists'])["user_id"].count().reset_index().sort_values(['user_id'],
                                                                                                         ascending=False)
    top_listening_songs.columns = ["song_name", "artists", "total_likes"]
    print(top_listening_songs[:10])


def collaboration_filtering_recommendation(song_name):
    # Model-based approach. Model is based on user ratings.
    # KNN algorithm is used for recommendations.
    # Improvements: Matrix Factorization - Decompose the original sparse user-item matrix into lower dimensionality
    #                                      less sparse rectangular matrices
    print("\n Collaborative Filtering Recommendation Started.. \n\n")
    tracks_dataset = data_util.load_data(FILENAME, True)
    liked_songs_dataset = data_util.load_data("data/liked_tracks.csv", True, ['user_id', 'track_id', "listen_count"])
    merged_datasets = pd.merge(left=liked_songs_dataset, right=tracks_dataset, how="inner", on="track_id")
    print("Length of merged dataset: ", len(merged_datasets))
    # data_exploration(merge_datasets)
    print(merged_datasets.head())

    # convert the dataframe into a pivot table
    df_songs_features = merged_datasets.pivot(index='track_id', columns='user_id', values='listen_count').fillna(0)
    print("\nPivot table: \n", df_songs_features.head(1))

    # obtain a sparse matrix
    mat_songs_features = csr_matrix(df_songs_features.values)
    print("\nCSR Matrix: ", mat_songs_features[:10])

    df_unique_tracks = tracks_dataset.drop_duplicates(subset=['track_id']).reset_index(drop=True)[['track_id', 'name']]
    print("\n Unique tracks length: ", len(df_unique_tracks))
    print(df_unique_tracks.head())

    decode_id_song = {
        song: i for i, song in
        enumerate(list(df_unique_tracks.set_index('track_id').loc[df_songs_features.index].name))
    }
    model = collaborative_recommender.Recommender(metric='cosine',
                                                  algorithm='brute',
                                                  k=20,
                                                  data=mat_songs_features,
                                                  decode_id_song=decode_id_song)
    new_recommendations = model.make_recommendation(new_song=song_name, n_recommendations=5)
    print("\n Collaboration filter Recommendations: ")
    for recommendation in new_recommendations:
        print(f"\t{recommendation}")


if __name__ == "__main__":
    # data_util.generate_random_user_likes(data_util.load_data(FILENAME, True), "data/liked_tracks.csv", 20, 1000)
    print("Processing started..")
    data = data_util.load_data(FILENAME)
    content_based_recommendation(data)
    collaboration_filtering_recommendation("Suspicious Minds")
    print("Processing finished..")

