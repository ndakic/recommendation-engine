import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from article.model_evaluator import ModelEvaluator
from article.cb_recommender import ContentBasedRecommender
from article.cf_recommender import CFRecommender
from article.hybrid_recommender import HybridRecommender

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0,
   'BOOKMARK': 2.5,
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,
}


def inspect_interactions(person_id, interactions_train_indexed_df, interactions_test_indexed_df, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df

    return interactions_df.loc[person_id].merge(articles_df,
                                                how='left',
                                                left_on='contentId',
                                                right_on='contentId') \
                          .sort_values('eventStrength', ascending=False)[['eventStrength', 'contentId', 'title', 'url', 'lang']]


def smooth_user_preference(x):
    return math.log(1+x, 2)


def data_munging(interactions):
    interactions['eventStrength'] = interactions['eventType'].apply(lambda x: event_type_strength[x])
    users_interactions_count_df = interactions.groupby(['personId', 'contentId']).size().groupby('personId').size()
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
    # number of interactions from users with at least 5 interactions
    interactions_from_selected_users_df = interactions.merge(users_with_enough_interactions_df,
                                                                how='right',
                                                                left_on='personId',
                                                                right_on='personId')
    # number of unique user/item interactions
    interactions_full_df = interactions_from_selected_users_df \
        .groupby(['personId', 'contentId'])['eventStrength'].sum() \
        .apply(smooth_user_preference).reset_index()

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)

    print('# interactions on Train set: %d' % len(interactions_train_df))
    print('# interactions on Test set: %d' % len(interactions_test_df))
    print(interactions_train_df.head(5))

    return interactions_full_df, interactions_train_df, interactions_test_df


if __name__ == "__main__":
    articles_df = pd.read_csv('data/shared_articles.csv')
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    interactions_df = pd.read_csv('data/users_interactions.csv')

    interactions_full_df, interactions_train_df, interactions_test_df = data_munging(interactions_df)
    print(interactions_full_df[interactions_full_df["personId"] == -1479311724257856983])
    model_evaluator = ModelEvaluator(articles_df, interactions_full_df, interactions_train_df, interactions_test_df)

    cb_recommender_model = ContentBasedRecommender(articles_df, interactions_train_df)
    print('Evaluating Content-Based Filtering model...')
    cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(cb_recommender_model)
    print('\nGlobal metrics:\n%s' % cb_global_metrics)

    cf_recommender_model = CFRecommender(articles_df, interactions_train_df)
    cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
    print('\nGlobal metrics:\n%s' % cf_global_metrics)
    print(cb_detailed_results_df.head(10))
    print(cf_detailed_results_df.head(10))

    hybrid_recommender_model = HybridRecommender(cb_recommender_model, cf_recommender_model, articles_df,
                                                 cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)
    print('Evaluating Hybrid model...')
    hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
    print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
    hybrid_detailed_results_df.head(10)

    global_metrics_df = pd.DataFrame([cb_global_metrics, cf_global_metrics, hybrid_global_metrics]).set_index('modelName')
    print(global_metrics_df)

    ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15, 8))
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')
    plt.show()

    print(hybrid_recommender_model.recommend_items(-1479311724257856983, topn=20, verbose=True))
