import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment-Based Product Recommendation",
    page_icon="🛍️",
    layout="centered"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    train             = joblib.load('dataset.gz')
    mnb               = joblib.load('mnb.gz')
    user_final_rating = joblib.load('user_final_rating.gz')
    vectorizer        = joblib.load('vectorizer.gz')
    return train, mnb, user_final_rating, vectorizer

train, mnb, user_final_rating, vectorizer = load_models()

# ── Recommendation logic (same as your model.py) ─────────────────────────────
def get_sentiment_recommendations(user):
    if user in user_final_rating.index:
        recommendations = list(
            user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        )
        temp = train[train.name.isin(recommendations)].copy()
        X = vectorizer.transform(temp["reviews_clean"].values.astype(str))
        temp["predicted_sentiment"] = mnb.predict(X)
        temp = temp[['name', 'predicted_sentiment']]
        temp_grouped = temp.groupby('name', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(
            lambda x: temp[(temp.name == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count()
        )
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(
            temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2
        )
        temp_grouped.drop('predicted_sentiment', axis=1, inplace=True)
        sorted_top_5 = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
        top_5_products = pd.merge(
            train[['name', 'brand', 'manufacturer']].drop_duplicates(),
            sorted_top_5[['name', 'pos_sentiment_percent']],
            on='name'
        ).sort_values('pos_sentiment_percent', ascending=False).rename(columns={
            'pos_sentiment_percent': 'Positive Sentiment %',
            'name': 'Product Name',
            'brand': 'Brand',
            'manufacturer': 'Manufacturer'
        }).reset_index(drop=True)
        top_5_products.index = np.arange(1, len(top_5_products) + 1)
        return top_5_products
    else:
        return None

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛍️ Sentiment-Based Product Recommendation")
st.markdown("Enter a username to get **top 5 product recommendations** based on sentiment analysis of reviews.")

st.markdown("---")

user_input = st.text_input("👤 Enter Username", placeholder="e.g. 00sab00")

if st.button("Get Recommendations", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a username.")
    else:
        with st.spinner("Analysing reviews and generating recommendations..."):
            result = get_sentiment_recommendations(user_input.strip())

        if result is None:
            st.error(f"❌ User **'{user_input}'** not found in the system. Please try a different username.")
        else:
            st.success(f"✅ Top 5 recommendations for **{user_input}**:")
            st.dataframe(result, use_container_width=True)

st.markdown("---")
st.markdown(
    "<small>Built by **Abhradeep Chandra Paul** · "
    "[GitHub](https://github.com/abhra-deep) · "
    "[LinkedIn](https://linkedin.com/in/abhradeepchandrapaul)</small>",
    unsafe_allow_html=True
)
