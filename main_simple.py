import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Step 1: Set up Spotify API access
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIPY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI'),
    scope="playlist-modify-private playlist-read-private"
))

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

playlist_id = 'https://open.spotify.com/playlist/5W7jVgr3X0yveoiOyIzvZF?si=c2f6218df1344616'  # Replace with your actual playlist ID
tracks = get_playlist_tracks(playlist_id)
track_ids = [track['track']['id'] for track in tracks]
audio_features = get_audio_features(track_ids)

# Step 3: Preprocess the data
feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'duration_ms',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

# Filter out tracks with no audio features
valid_tracks = [(i, track) for i, track in enumerate(audio_features) if track is not None]
valid_indices, valid_features = zip(*valid_tracks)

X = np.array([[track[feature] for feature in feature_names] for track in valid_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Perform clustering
n_clusters = 5  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

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
    cluster_tracks[label].append(track_ids[valid_indices[i]])

# Set the fixed number of songs per playlist
songs_per_playlist = 50  # Adjust this number as needed

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