#############################
# Content Based Recommendation
#############################

#############################
# Developing Recommendations Based on Movie Reviews
#############################

#################################
# 1. Create TF-IDF Matrix
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # Disabling DtypeWarning
df.head()
df.shape

# View the 'overview' column of the dataset
df["overview"].head()

# Initialize the TF-IDF Vectorizer, excluding common English stop words
tfidf = TfidfVectorizer(stop_words="english")

# Fill missing values in the 'overview' column with an empty string
df['overview'] = df['overview'].fillna('')

# Create the TF-IDF matrix from the 'overview' column
tfidf_matrix = tfidf.fit_transform(df['overview'])

# View the shape of the TF-IDF matrix
tfidf_matrix.shape

# Check the number of movie titles
df['title'].shape

# Get the feature names (words)
tfidf.get_feature_names()

# Convert the TF-IDF matrix to an array
tfidf_matrix.toarray()

#################################
# 2. Creation of the Cosine Similarity Matrix
#################################

# Calculate the cosine similarity between movies based on their overview
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# View the shape of the cosine similarity matrix
cosine_sim.shape

# View the cosine similarity scores for the second movie
cosine_sim[1]

#################################
# 3. Making Recommendations Based on Similarities
#################################

# Create a series with movie titles as the index
indices = pd.Series(df.index, index=df['title'])

# Check how many times each title appears in the dataset
indices.index.value_counts()

# Remove duplicated titles, keeping only the last one
indices = indices[~indices.index.duplicated(keep='last')]

# Find the index for a specific movie title
indices["Cinderella"]

indices["Sherlock Holmes"]

# Get the index of the movie 'Sherlock Holmes'
movie_index = indices["Sherlock Holmes"]

# View the cosine similarity scores for 'Sherlock Holmes'
cosine_sim[movie_index]

# Create a DataFrame of the similarity scores for the movie
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

# Sort the movies based on similarity scores and get the top 10 most similar movies (excluding the movie itself)
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# Get the movie titles of the top 10 most similar movies
df['title'].iloc[movie_indices]

#################################
# 4. Preparing the Script
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    """
    Recommender function that returns the top 10 similar movies based on a given title.
    
    Args:
    - title (str): The title of the movie to find recommendations for.
    - cosine_sim (array): The cosine similarity matrix of movie overviews.
    - dataframe (pd.DataFrame): The dataframe containing movie data.
    
    Returns:
    - pd.Series: A list of the top 10 most similar movie titles.
    """
    # Create a series with movie titles as the index
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    
    # Get the index of the movie based on the title
    movie_index = indices[title]
    
    # Get the similarity scores for the movie
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    
    # Get the top 10 most similar movies (excluding the movie itself)
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    
    return dataframe['title'].iloc[movie_indices]

# Test the content-based recommender function
content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Matrix", cosine_sim, df)
content_based_recommender("The Godfather", cosine_sim, df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

def calculate_cosine_sim(dataframe):
    """
    Function to calculate the cosine similarity matrix for movie overviews.
    
    Args:
    - dataframe (pd.DataFrame): The dataframe containing movie data.
    
    Returns:
    - array: The cosine similarity matrix.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Fill missing values in the 'overview' column
    dataframe['overview'] = dataframe['overview'].fillna('')
    
    # Create the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    
    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

# Calculate the cosine similarity matrix for the movie dataset
cosine_sim = calculate_cosine_sim(df)

# Get movie recommendations based on 'The Dark Knight Rises'
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
