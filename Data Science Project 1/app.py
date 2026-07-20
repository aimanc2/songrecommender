import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
final_df = pd.read_csv(os.path.join(BASE_DIR, 'streamlit datasets', 'final_matched_catalog - Copy.csv'))
scaled_df = pd.read_csv(os.path.join(BASE_DIR, 'streamlit datasets', 'scaled_df - Copy.csv'))

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.title("Song Recommender")
st.write("Pick a few songs you like, and get similar recommendations.")

with st.expander("How does this work?"):
    st.write("""
    Each song is represented by 9 audio features (danceability, energy, tempo, etc.), 
    scaled to a comparable range. When you pick songs you like, the app averages their 
    features into a single "taste profile."

    You will see a similarity score for each recommended song. A score close to 1 means very similar 
    sound characteristics; a score closer to 0 and further from 1 means less similarity.

    The recommendations you see are simply the catalog's songs ranked by that similarity 
    score, from most to least alike.
    """)

# Build a display list like "Artist - Title" so songs are identifiable
final_df['display'] = final_df['Artist'] + " - " + final_df['Title']

# Let the user pick songs from your catalog
selected = st.multiselect("Choose songs you like:", final_df['display'].tolist())

if st.button("Get Recommendations"):
    if len(selected) == 0:
        st.warning("Pick at least one song first.")
    else:
        # Find the index of the songs the user picked
        liked_idx = final_df[final_df['display'].isin(selected)].index

        # Build the user profile (average of liked songs' features)
        liked_vectors = scaled_df.loc[liked_idx]
        user_profile = liked_vectors.mean(axis=0)

        # Compute similarity against the whole catalog
        similarities = cosine_similarity(
            user_profile.values.reshape(1, -1), scaled_df
        )
        final_df['similarity'] = similarities[0]

        # Sort, exclude already-liked songs, show top 10
        recommendations = final_df.sort_values('similarity', ascending=False)
        recommendations = recommendations[~recommendations.index.isin(liked_idx)]

        st.subheader("Recommended for you:")
        st.dataframe(recommendations[['Artist', 'Title', 'similarity']].head(10))