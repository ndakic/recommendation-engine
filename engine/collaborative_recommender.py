from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz


class CFRecommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.model = self._recommender().fit(data)

    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)

    def _get_recommendations(self, new_song, n_recommendations):
        # Get the id of the song according to the text
        recommend_song_id = self._fuzzy_matching(song=new_song)
        # Start the recommendation process
        print(f"\nStarting the recommendation process for song: {new_song} (song_id: {recommend_song_id}) ...")
        # Return the n neighbors for the song id
        distances, indices = self.model.kneighbors(self.data[recommend_song_id], n_neighbors=n_recommendations)
        # print("song: ", self.data[recommend_song_id]) ---> (0, 753)	9.0
        # print("data: ", self.data)                    ---> (0, 670)	9.0, (1, 803)	1.0
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    def _map_indices_to_song_title(self):
        # get reverse mapper
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}

    def _fuzzy_matching(self, song):
        match_tuple = []
        # get match
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title.lower(), song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print(f"The recommendation system could not find a match for {song}")
            return
        return match_tuple[0][1]

    def _recommend(self, new_song, n_recommendations):
        # Get the id of the recommended songs
        recommendations = []
        recommendation_ids = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
        # return the name of the song using a mapping dictionary
        recommendations_map = self._map_indices_to_song_title()
        # Translate this recommendations into the ranking of song titles recommended
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations

    def make_recommendation(self, new_song, n_recommendations):
        recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
        return recommended
