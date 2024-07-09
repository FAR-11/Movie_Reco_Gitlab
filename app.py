import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge movies and ratings datasets
data = pd.merge(ratings, movies, on='movieId')

# Create user-item interaction matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Fit KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix_filled.values)

# Recommendation function
def recommend_movies(user_idx, data, model, n_recommendations):
    try:
        user_data = data.iloc[user_idx, :].values.reshape(1, -1)
        st.write(f"User data shape: {user_data.shape}")
        distances, indices = model.kneighbors(user_data, n_neighbors=n_recommendations+1)
        st.write(f"Distances: {distances}")
        st.write(f"Indices: {indices}")
        recommendations = []
        for i in range(1, len(indices.flatten())):
            movie_title = data.columns[indices.flatten()[i]]
            recommendations.append(movie_title)
        return recommendations
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Streamlit app
st.title('Movie Recommendation System')
user_id = st.number_input('Enter User ID', min_value=1, max_value=user_movie_matrix_filled.shape[0])
n_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=10, value=5)

if st.button('Get Recommendations'):
    if user_id in user_movie_matrix_filled.index:
        user_idx = user_movie_matrix_filled.index.get_loc(user_id)
        st.write(f"User index: {user_idx}")
        st.write(f"User movie matrix index: {user_movie_matrix_filled.index}")
        st.write(f"User movie matrix columns: {user_movie_matrix_filled.columns}")
        recommendations = recommend_movies(user_idx, user_movie_matrix_filled, model_knn, n_recommendations)
        if recommendations:
            st.write(f'Recommended movies for user {user_id}:')
            for rec in recommendations:
                st.write(rec)
        else:
            st.write('No recommendations could be made.')
    else:
        st.error('User ID not found in the dataset.')