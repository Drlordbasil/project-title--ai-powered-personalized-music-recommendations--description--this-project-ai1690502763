import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

# Step 1: Load and preprocess user data
users_data = pd.read_csv('user_data.csv')  # Assuming user data is stored in a CSV file

# Step 2: User Profiling
def build_user_profile(user_id):
    user_history = users_data[users_data['user_id'] == user_id]['listening_history']
    user_genre_preferences = users_data[users_data['user_id'] == user_id]['genre_preferences']
    user_ratings = users_data[users_data['user_id'] == user_id]['song_ratings']
    # Build personalized profile using user's listening history, genre preferences, and song ratings
    user_profile = {
        'user_id': user_id,
        'listening_history': user_history,
        'genre_preferences': user_genre_preferences,
        'song_ratings': user_ratings
    }
    return user_profile

# Step 3: Collaborative Filtering
def collaborative_filtering(user_profile):
    # Build user-to-song matrix using user ratings
    user_ratings_matrix = users_data.pivot(index='user_id', columns='song_id', values='song_ratings').fillna(0)
    # Compute cosine similarity between users based on their ratings
    similarity_matrix = cosine_similarity(user_ratings_matrix, user_ratings_matrix)
    # Retrieve similar users with highest similarity scores
    similar_users = similarity_matrix[user_profile['user_id']].argsort()[:-5:-1]
    # Extract top recommendations from similar users' listening history
    top_recommendations = users_data[users_data['user_id'].isin(similar_users)]['listening_history']
    return top_recommendations

# Step 4: Content-Based Filtering
def content_based_filtering(user_profile):
    # Extract genre preferences
    user_genre_preferences = user_profile['genre_preferences']
    # Combine genre preferences into a single string
    genre_text = ' '.join(user_genre_preferences)
    # Calculate term frequency-inverse document frequency (TF-IDF) for songs' genre tags
    tfidf = TfidfVectorizer()
    song_genre_matrix = tfidf.fit_transform(users_data['genre_tag'])
    # Compute cosine similarity between user's genre preferences and songs' genre tags
    similarity_scores = cosine_similarity(tfidf.transform([genre_text]), song_genre_matrix)
    # Retrieve top recommendations based on genre similarity
    top_recommendations = users_data.iloc[similarity_scores.argsort()[0]][::-1]['listening_history']
    return top_recommendations

# Step 5: Hybrid Approach
def hybrid_approach(user_profile):
    collaborative_recommendations = collaborative_filtering(user_profile)
    content_based_recommendations = content_based_filtering(user_profile)
    # Combine both recommendation lists for comprehensive suggestions
    hybrid_recommendations = collaborative_recommendations.append(content_based_recommendations).drop_duplicates()
    return hybrid_recommendations

# Step 6: Real-Time Updates and User Feedback
def update_recommendations(user_profile, feedback):
    # Update user's listening history, genre preferences, or song ratings based on feedback
    # Re-run the recommendation process with the updated user profile
    updated_user_profile = update_user_profile(user_profile, feedback)
    updated_recommendations = hybrid_approach(updated_user_profile)
    return updated_recommendations

# Step 7: Web Application Interface using Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = request.form['user_id']
    user_profile = build_user_profile(user_id)
    user_feedback = request.form['feedback']
    updated_recommendations = update_recommendations(user_profile, user_feedback)
    return render_template('recommendations.html', recommendations=updated_recommendations)

if __name__ == '__main__':
    app.run(debug=True)

# Note: This is a simplified version of the program. Implementation details and functionalities may vary depending on specific requirements and dataset availability.