import os
import csv
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lyricsgenius
import re
import pandas as pd
import time
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

# Load environment variables from .env file
load_dotenv()

# Function to create Spotify client with retry logic
def create_spotify_client(max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            return spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=os.getenv('SPOTIPY_CLIENT_ID'),
                client_secret=os.getenv('SPOTIPY_CLIENT_SECRET'),
                redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI'),
                scope="playlist-modify-private playlist-read-private"
            ))
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt < max_retries - 1:
                print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect after {max_retries} attempts. Please check your internet connection and try again later.")
                raise

# Create Spotify client
sp = create_spotify_client()

# Set up Genius API access
genius = lyricsgenius.Genius(os.getenv('GENIUS_ACCESS_TOKEN'))

# Function to delete existing cluster playlists
def delete_existing_cluster_playlists():
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        if playlist['name'].startswith("(o_o) ") and playlist['name'].endswith(" Playlist"):
            print(f"Deleting existing playlist: {playlist['name']}")
            sp.current_user_unfollow_playlist(playlist['id'])

# Step 2: Retrieve track features for your playlist
def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def get_audio_features(track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 100):
        audio_features.extend(sp.audio_features(track_ids[i:i+100]))
    return audio_features

# Function to fetch lyrics using Genius API
def get_lyrics(track_name, artist_name):
    try:
        song = genius.search_song(track_name, artist_name)
        if song:
            # Clean up lyrics: remove verse headers, empty lines, and extra whitespace
            lyrics = re.sub(r'\[.*?\]|\n\s*\n', '\n', song.lyrics)
            cleaned_lyrics = lyrics.strip()
            # Check if the cleaned lyrics are empty or just contain common filler words
            if not cleaned_lyrics or cleaned_lyrics.lower() in ['instrumental', 'instumental', 'no lyrics']:
                return "empty"
            return cleaned_lyrics
        else:
            # If no lyrics found, it might be an instrumental track
            return "empty"
    except Exception as e:
        print(f"Error fetching lyrics for {track_name} by {artist_name}: {str(e)}")
        return "error"

# Function to save song data to CSV
def save_to_csv(song_data, filename='song_data.csv'):
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=song_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(song_data)
        return True
    except Exception as e:
        print(f"Error saving data for {song_data.get('name', 'Unknown')}: {str(e)}")
        return False

# Function to load processed songs from CSV
def load_processed_songs(filename='song_data.csv'):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        if 'id' not in df.columns:
            print("Warning: 'id' column not found in the CSV. Starting with an empty set of processed songs.")
            return pd.DataFrame(columns=['id'])
        return df
    print("CSV file not found. Starting with an empty DataFrame.")
    return pd.DataFrame(columns=['id'])

# Function to process a single track
def process_track(track, features):
    track_id = track['track']['id']
    track_name = track['track']['name']
    artist_name = track['track']['artists'][0]['name']
    
    # Get lyrics
    lyrics = get_lyrics(track_name, artist_name)
    
    # Prepare song data
    song_data = {
        'id': track_id,
        'name': track_name,
        'artist': artist_name,
        'lyrics': lyrics
    }
    
    # Add audio features
    if features:
        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'duration_ms',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']:
            song_data[feature] = features[feature]
    
    return song_data

def get_tracks_with_retry(playlist_id, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            tracks = get_playlist_tracks(playlist_id)
            track_ids = [track['track']['id'] for track in tracks]
            audio_features = get_audio_features(track_ids)
            return tracks, audio_features
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt < max_retries - 1:
                print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to fetch tracks after {max_retries} attempts. Please check your internet connection and try again later.")
                raise

# Main script
if __name__ == "__main__":
    try:
        playlist_id = 'https://open.spotify.com/playlist/5W7jVgr3X0yveoiOyIzvZF?si=c2f6218df1344616'  # Replace with your actual playlist ID
        tracks, audio_features = get_tracks_with_retry(playlist_id)

        # Load already processed songs
        processed_songs = load_processed_songs()
        processed_ids = set(processed_songs['id'].tolist() if not processed_songs.empty else [])

        # Step 3: Process tracks and update CSV
        print("Processing tracks and updating CSV... This may take a while.")
        for track, features in zip(tracks, audio_features):
            track_id = track['track']['id']
            
            # Skip if already processed
            if track_id in processed_ids:
                continue
            
            try:
                song_data = process_track(track, features)
                if save_to_csv(song_data):
                    print(f"Processed and saved data for {song_data['name']} by {song_data['artist']} (Lyrics: {'Empty' if song_data['lyrics'] == 'empty' else 'Found'})")
                    processed_ids.add(track_id)
                else:
                    print(f"Failed to save data for {song_data['name']} by {song_data['artist']}")
            except Exception as e:
                print(f"Error processing track {track['track']['name']}: {str(e)}")
            
            # Add a small delay to avoid hitting API rate limits
            time.sleep(1)

        # Load all processed songs
        all_songs = load_processed_songs()

        # Prepare features for clustering
        feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'duration_ms',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

        X = all_songs[feature_names].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Process lyrics
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        lyrics_features = tfidf_vectorizer.fit_transform(all_songs['lyrics'].fillna(''))

        # Process artist names
        artist_vectorizer = TfidfVectorizer(max_features=50)
        artist_features = artist_vectorizer.fit_transform(all_songs['artist'])

        # Combine all features
        svd = TruncatedSVD(n_components=10)
        lyrics_reduced = svd.fit_transform(lyrics_features)
        artist_reduced = svd.fit_transform(artist_features)

        X_combined = np.hstack((X_scaled, lyrics_reduced, artist_reduced))

        # Step 4: Perform clustering
        n_clusters = 5  # You can adjust this number
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_combined)

        # Step 5: Create new playlists based on clusters
        def create_playlist(name, track_ids):
            playlist = sp.user_playlist_create(sp.me()['id'], name, public=False)
            return playlist['id']

        def add_tracks_to_playlist(playlist_id, track_ids):
            # Add tracks in batches of 100
            for i in range(0, len(track_ids), 100):
                sp.playlist_add_items(playlist_id, track_ids[i:i+100])

        # Delete existing cluster playlists
        delete_existing_cluster_playlists()

        # Group tracks by cluster
        cluster_tracks = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_tracks[label].append(all_songs['id'].iloc[i])

        # Set the fixed number of songs per playlist
        songs_per_playlist = 25  # Adjust this number as needed

        # Create new cluster playlists with fixed number of songs
        for i in range(n_clusters):
            playlist_name = f"(o_o) {i+1} Playlist"
            playlist_tracks = cluster_tracks[i][:songs_per_playlist]  # Take only the first 'songs_per_playlist' tracks
            
            if len(playlist_tracks) < songs_per_playlist:
                print(f"Warning: Cluster {i+1} has only {len(playlist_tracks)} tracks.")
            
            playlist_id = create_playlist(playlist_name, playlist_tracks)
            add_tracks_to_playlist(playlist_id, playlist_tracks)
            print(f"Created playlist '{playlist_name}' with {len(playlist_tracks)} tracks. ID: {playlist_id}")

        print("Clustering and playlist creation complete!")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("The script will now exit. Please check the error message and try again.")