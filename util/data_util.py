import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_TYPES = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
COLUMN_NAMES = ["id", "name", "album", "album_id", "artists", "artist_ids", "track_number", "disc_number", "explicit",
                "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms", "time_signature", "year", "release_date"]


def generate_random_user_likes(dataset, target_file_name, total_number_of_tracks, total_number_of_users):
    results = []
    for i in range(0, total_number_of_users):
        random_tracks = dataset.sample(total_number_of_tracks)[["id"]]
        random_tracks = random_tracks.rename(columns={'id': 'track_id'})
        random_tracks['user_id'] = "user-{}".format(i+1)
        results.append(random_tracks)

    final_result = pd.concat(results)
    final_result.to_csv(target_file_name, columns=["user_id", "track_id"], index=False)
    return final_result


def print_random_samples(data):
    print("10 random samples from dataset: \n", data.sample(10))


def load_data(filename, raw=False, column_names=False):
    if not column_names:
        column_names = COLUMN_NAMES
    data = pd.read_csv(filename, skiprows=[0], sep=",", names=column_names, low_memory=False)
    if raw:
        return data
    selected_features = data.drop(columns=['id', 'release_date', 'year', 'album', 'album_id', 'artist_ids'])
    print("Loaded data (10 samples): \n", selected_features.sample(10))
    return selected_features


def data_exploration(data):
    print("\nData info:")
    data.info()
    print("=====================================")
    print("Check correlation between features:\n")
    print(data.corr())
    print("=====================================")


def data_normalization(data):
    n_data = data.select_dtypes(include=DATA_TYPES)
    scaler = MinMaxScaler()
    data[n_data.columns] = scaler.fit_transform(n_data[n_data.columns])
