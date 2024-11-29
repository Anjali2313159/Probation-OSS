import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#sample
data = {
    'User': ['Animesh', 'Aishwarya', 'Anjali', 'Pari', 'Krit'],
    'Movie1': [5, 4, 0, 0, 2],
    'Movie2': [3, 0, 0, 5, 4],
    'Movie3': [0, 0, 4, 4, 0],
    'Movie4': [4, 3, 0, 0, 0],
    'Movie5': [0, 5, 0, 3, 0],
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('User', inplace=True)

# Function to recommend movies
def recommend_movies(user, df, num_recommendations=2):
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(df.fillna(0))
    
    # Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    
    # Get the user's similarity scores
    user_similarity = similarity_df[user].sort_values(ascending=False)
    
    # Get the top similar users
    top_users = user_similarity.index[1:]  
    
    # Create a series to hold movie recommendations
    recommendations = pd.Series(dtype=float)
    
    # Iterate through the top users
    for top_user in top_users:
        # Get the movies rated by the top user
        rated_movies = df.loc[top_user]
        
        # Get the user's ratings
        user_ratings = df.loc[user]
        
        # Recommend movies not rated by the user
        for movie, rating in rated_movies.items():
            if user_ratings[movie] == 0: 
                recommendations[movie] = recommendations.get(movie, 0) + rating
    
    # Sort recommendations by score
    recommendations = recommendations.sort_values(ascending=False)
    
    return recommendations.head(num_recommendations)

# Example 
user_to_recommend = 'Animesh'
recommendations = recommend_movies(user_to_recommend, df)
print(f"Movie recommendations for {user_to_recommend}:")
print(recommendations)