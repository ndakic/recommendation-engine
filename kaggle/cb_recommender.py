import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from kaggle.user_profile import UserProfile


def prepare_data(articles_df, interactions_train_df):
    # Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
    stopwords_list = stopwords.words('english') + stopwords.words('portuguese')
    # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.003,
                                 max_df=0.5,
                                 max_features=5000,
                                 stop_words=stopwords_list)
    item_ids = articles_df['contentId'].tolist()
    tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
    user_profiles = UserProfile(articles_df, interactions_train_df, tfidf_matrix, item_ids).build_users_profiles()
    return tfidf_matrix, item_ids, user_profiles


class ContentBasedRecommender:

    MODEL_NAME = 'Content-Based'

    def __init__(self, articles_df, interactions_train_df):
        self.tfidf_matrix, self.item_ids, self.user_profiles = prepare_data(articles_df, interactions_train_df)
        self.items_df = articles_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']).head(topn)
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            recommendations_df = recommendations_df.merge(self.items_df,
                                                          how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]
        return recommendations_df
