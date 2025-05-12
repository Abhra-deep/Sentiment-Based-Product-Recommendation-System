import joblib
import pandas as pd
import numpy as np

# loading required objects
train = joblib.load('dataset.gz')
mnb = joblib.load('mnb.gz')
user_final_rating = joblib.load('user_final_rating.gz')
vectorizer = joblib.load('vectorizer.gz')
def get_sentiment_recommendations(user):
    """
    Function to get top 5 product recommendations for a user,
    enhanced by predicted sentiment of reviews using a classifier.

    Steps:
    - Check if user exists in recommendation matrix.
    - Extract top 20 recommended products for the user.
    - Predict sentiment for all reviews of these products.
    - Group sentiment by product and calculate positive sentiment %.
    - Merge with product details and return top 5 products.
    """

    if (user in user_final_rating.index):
        # Step 1: Get top 20 product recommendations for the user
        recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

        # Step 2: Filter reviews of these recommended products from the training data
        temp = train[train.name.isin(recommendations)].copy()

        # Step 3: Transform review text using trained vectorizer
        X = vectorizer.transform(temp["reviews_clean"].values.astype(str))

        # Step 4: Predict sentiment using trained classifier
        temp["predicted_sentiment"] = mnb.predict(X)

        # Step 5: Keep only relevant columns for sentiment aggregation
        temp = temp[['name', 'predicted_sentiment']]

        # Step 6: Count number of reviews per product
        temp_grouped = temp.groupby('name', as_index=False).count()

        # Step 7: Count number of positive reviews per product
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(
            lambda x: temp[(temp.name == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count()
        )

        # Step 8: Store total number of reviews per product
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']

        # Step 9: Calculate percentage of positive sentiment reviews
        temp_grouped['pos_sentiment_percent'] = np.round(
            temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2
        )

        # Step 10: Drop intermediate column used for counting
        temp_grouped.drop('predicted_sentiment', axis=1, inplace=True)

        # Step 11: Sort products by positive sentiment percentage and take top 5
        sorted_top_5 = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]

        # Step 12: Merge top 5 products with brand and manufacturer details
        top_5_products = pd.merge(
            train[['name', 'brand', 'manufacturer']].drop_duplicates(),
            sorted_top_5[['name', 'pos_sentiment_percent']],
            on='name'
        ).sort_values('pos_sentiment_percent', ascending=False).rename(
            columns={
                'pos_sentiment_percent': 'Positive Sentiment %',
                'name': 'Name',
                'brand': 'Brand',
                'manufacturer': 'Manufacturer'
            }
        ).reset_index(drop=True)

        # Step 13: Reformat index for better presentation
        top_5_products.index = np.arange(1, len(top_5_products) + 1)
        top_5_products.columns.name = 'S. No.'

        return top_5_products

    else:
        # If the user does not exist in the recommendation matrix
        print(f"User name {user} doesn't exist")


def dataframe_to_html(df):
    """
    Convert the dataframe into a nice presentable table for the end user
    """
    html_resp = df.to_html().replace(
        'table border="1"', 'table border="1" style="border-collapse:collapse"'
    ).replace(
        'tr style="text-align: right;"', 'tr style="text-align: center; background-color: beige;"'
    ).replace(
        '<td>', '<td style="text-align:center; padding: 0.5em;">'
    ).replace(
        '<th>', '<th style="text-align:center; padding: 0.3em;">'
    )

    return html_resp


if __name__ == "__main__":
    # user does not exist
    print(get_sentiment_recommendations('does-not-exist'))

    # user exists
    print(get_sentiment_recommendations('00sab00'))