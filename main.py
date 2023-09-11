# Optimize this Python script:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
from typing import List

app = Flask(__name__)

# Step 1: Load and preprocess user data
# Assuming user data is stored in a CSV file
users_data = pd.read_csv('user_data.csv')


# Step 2: User Profiling
def build_user_profile(user_id: int) -> dict:
    user_data = users_data[users_data['user_id'] == user_id]
    user_profile = user_data[['user_id', 'listening_history',
                              'genre_preferences', 'song_ratings']].to_dict(orient='records')[0]
    return user_profile


# Step 3: Collaborative Filtering
def collaborative_filtering(user_profile: dict) -> pd.Series:
    user_ratings_matrix = users_data.pivot(
        index='user_id', columns='song_id', values='song_ratings').fillna(0)
    similarity_matrix = cosine_similarity(
        user_ratings_matrix, user_ratings_matrix)
    similar_users = similarity_matrix[user_profile['user_id']].argsort()[
        :-5:-1]
    top_recommendations = users_data[users_data['user_id'].isin(
        similar_users)]['listening_history'].squeeze()
    return top_recommendations


# Step 4: Content-Based Filtering
def content_based_filtering(user_profile: dict) -> pd.Series:
    user_genre_preferences = user_profile['genre_preferences']
    genre_text = ' '.join(user_genre_preferences)
    tfidf = TfidfVectorizer()
    song_genre_matrix = tfidf.fit_transform(users_data['genre_tag'])
    similarity_scores = cosine_similarity(
        tfidf.transform([genre_text]), song_genre_matrix)
    top_recommendations = users_data.iloc[similarity_scores.argsort()[
        0]][::-1].iloc[:, 2].squeeze()
    return top_recommendations


# Step 5: Hybrid Approach
def hybrid_approach(user_profile: dict) -> pd.Series:
    collaborative_recommendations = collaborative_filtering(user_profile)
    content_based_recommendations = content_based_filtering(user_profile)
    hybrid_recommendations = collaborative_recommendations.append(
        content_based_recommendations).drop_duplicates()
    return hybrid_recommendations


# Step 6: Real-Time Updates and User Feedback
def update_recommendations(user_profile: dict, feedback: List[str]) -> pd.Series:
    updated_user_profile = user_profile.copy()
    updated_user_profile['listening_history'] += feedback
    updated_recommendations = hybrid_approach(updated_user_profile)
    return updated_recommendations


# Step 7: Web Application Interface using Flask
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = int(request.form['user_id'])
    user_profile = build_user_profile(user_id)
    user_feedback = request.form.getlist('feedback')
    updated_recommendations = update_recommendations(
        user_profile, user_feedback)
    return render_template('recommendations.html', recommendations=updated_recommendations)


if __name__ == '__main__':
    app.run(debug=True)

# Note: This is a simplified version of the program. Implementation details and functionalities may vary depending on specific requirements and dataset availability.
