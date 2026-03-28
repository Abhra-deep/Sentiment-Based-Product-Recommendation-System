# Sentiment-Based Product Recommendation System

A machine learning web application that recommends top 5 products to users based on sentiment analysis of customer reviews, built using Python, Scikit-learn, and Streamlit.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen)](https://sentiment-based-appuct-recommendation-system-abhra-deep.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-abhra--deep-blue)](https://github.com/abhra-deep/Sentiment-Based-Recommendation)
![](https://img.shields.io/badge/Maintained-Yes-indigo)

---

## 📌 About The Project

In e-commerce, recommending the right product is critical. This project builds an end-to-end NLP + ML pipeline that:

- Analyses 100K+ customer reviews using sentiment classification
- Recommends top 20 products per user using collaborative filtering
- Re-ranks those 20 products by positive sentiment % to return the best 5
- Displays results in a clean, interactive web interface

Built as part of my AI/ML portfolio to demonstrate real-world NLP and recommendation system deployment skills.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://sentiment-based-appuct-recommendation-system-abhra-deep.streamlit.app/)

### How to use:
1. Enter one of the sample usernames below
2. Click **"Get Recommendations"**
3. See your top 5 personalized product recommendations with sentiment scores!

### 👤 Sample Usernames to Try:
| Username |
|---|
| `00sab00` |
| `genius` |
| `frank` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Sentiment Model | Multinomial Naive Bayes (Scikit-learn) |
| Vectorizer | TF-IDF |
| Recommendation | User-Based Collaborative Filtering |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📊 How It Works

```
User Input (username)
        ↓
Collaborative Filtering → Top 20 Products
        ↓
Filter Reviews of those 20 Products
        ↓
TF-IDF Vectorization of Review Text
        ↓
Naive Bayes Sentiment Prediction (Positive / Negative)
        ↓
Rank Products by Positive Sentiment %
        ↓
Return Top 5 Products ✅
```

---

## 🧠 Model Details

- **Sentiment Classifier:** Multinomial Naive Bayes
- **Text Features:** TF-IDF Vectorizer on cleaned review text
- **Recommendation Base:** User-based collaborative filtering matrix
- **Output:** Top 5 products ranked by % positive sentiment reviews
- **Precision:** ~95%+ on sentiment classification

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/abhra-deep/Sentiment-Based-Recommendation.git

# Go into the folder
cd Sentiment-Based-Recommendation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Sentiment-Based-Recommendation/
│
├── app.py                    # Streamlit web app
├── model.py                  # Recommendation + sentiment logic
├── mnb.gz                    # Trained Naive Bayes model
├── vectorizer.gz             # Trained TF-IDF vectorizer
├── dataset.gz                # Cleaned training dataset
├── user_final_rating.gz      # User-product collaborative filtering matrix
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📬 Contact

**Abhradeep Chandra Paul**
- 📧 abhradeepchandrapaul@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/abhradeepchandrapaul)
- 🐙 [GitHub](https://github.com/abhra-deep)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">If you found this useful, please ⭐ the repo!</div>
