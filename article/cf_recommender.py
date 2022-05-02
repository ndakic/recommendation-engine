import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'
    NUMBER_OF_FACTORS_MF = 15

    def __init__(self, articles, interactions_train_df):

        # Creating a sparse pivot table with users in rows and items in columns
        users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                                  columns='contentId',
                                                                  values='eventStrength').fillna(0)

        users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
        users_ids = list(users_items_pivot_matrix_df.index)
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        u, sigma, v_t = svds(users_items_pivot_sparse_matrix, k=self.NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(u, sigma), v_t)
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                    all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        # Converting the reconstructed matrix back to a Pandas dataframe
        cf_predictions_df = pd.DataFrame(all_user_predicted_ratings_norm,
                                         columns=users_items_pivot_matrix_df.columns,
                                         index=users_ids).transpose()

        self.cf_predictions_df = cf_predictions_df
        self.items_df = articles

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df
