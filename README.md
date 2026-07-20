# Song Recommendation System

A content-based song recommender built with Python, Pandas, and Scikit-learn. 
Users pick a few songs they like, and the app recommends similar songs based 
on audio feature similarity (cosine similarity), not collaborative/user-history data.

## How it works
- Song catalog sourced from kworb.net (iTunes/Spotify charts)
- Audio features (danceability, energy, valence, tempo, etc.) sourced from a 
  pre-collected Kaggle dataset, matched by artist + title
- User taste profile = average of liked songs' scaled audio features
- Recommendations ranked by cosine similarity to that profile

## Run it locally

First download the entire 'Data Science Project 1' and open it in your Visual Studio Code.
Then run this command in the terminal file of app.py:
\```
pip install -r requirements.txt
streamlit run app.py
\```

## Limitations
- Content-based, not collaborative — no real user listening history was available
- ~54% of the chart catalog matched to the audio-features dataset; the rest 
  were excluded due to dataset coverage gaps
