# app.py
import streamlit as st
import pickle
import pandas as pd
import requests
from functools import lru_cache
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🍿",
    layout="wide"
)

# ------------------------------
# Session State Initialization
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # Stores imdb_id of recently viewed movies
if "mode" not in st.session_state:
    st.session_state.mode = None
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "random_movie" not in st.session_state:
    st.session_state.random_movie = None

# ------------------------------
# TMDB API and Helper Functions
# ------------------------------
# Expect a .streamlit/secrets.toml with:
# [tmdb]
# api_key = "YOUR_TMDB_API_KEY"
TMDB_API_KEY = st.secrets["tmdb"]["api_key"]

def requests_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

@lru_cache(maxsize=4096)
def _resolve_tmdb_id_from_imdb(imdb_id: str):
    """
    Resolve a TMDB numeric movie id from an imdb_id like 'tt1375666'.
    Returns int TMDB id or None.
    """
    if not imdb_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
        r = requests_retry_session().get(url, timeout=20)
        if r.status_code == 200:
            payload = r.json()
            results = payload.get("movie_results") or []
            if results:
                return results[0].get("id")
    except Exception as e:
        print("resolve_tmdb_id error:", e)
    return None

def fetch_poster_by_imdb(imdb_id: str):
    tmdb_id = _resolve_tmdb_id_from_imdb(imdb_id)
    if not tmdb_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        response = requests_retry_session().get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print("fetch_poster error:", e)
    return None

def fetch_trailer_by_imdb(imdb_id: str):
    tmdb_id = _resolve_tmdb_id_from_imdb(imdb_id)
    if not tmdb_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?api_key={TMDB_API_KEY}"
        response = requests_retry_session().get(url, timeout=20)
        if response.status_code == 200:
            for video in response.json().get("results", []):
                if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                    return f"https://youtu.be/{video['key']}"
    except Exception as e:
        print("fetch_trailer error:", e)
    return None

def get_movie_details_by_imdb(imdb_id: str, director_name: str = "N/A"):
    """
    Get details using imdb_id; we resolve to TMDB id then pull details.
    Director is taken from your dataframe column, so pass it in.
    """
    tmdb_id = _resolve_tmdb_id_from_imdb(imdb_id)
    if not tmdb_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos"
        response = requests_retry_session().get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()

            # Use your dataframe's 'director' value if provided
            directors = director_name if director_name else "N/A"

            # Cast (top 5 from TMDB credits)
            cast = data.get("credits", {}).get("cast", [])[:5]
            cast_details = []
            for actor in cast:
                cast_details.append({
                    "name": actor.get("name"),
                    "character": actor.get("character"),
                    "profile": f"https://image.tmdb.org/t/p/w500{actor['profile_path']}" if actor.get("profile_path") else None
                })

            genres = ", ".join([g["name"] for g in data.get("genres", [])]) if data.get("genres") else "N/A"
            budget = f"${data.get('budget', 0):,}" if data.get("budget", 0) > 0 else "N/A"
            revenue = f"${data.get('revenue', 0):,}" if data.get("revenue", 0) > 0 else "N/A"
            available_in = ", ".join([lang["english_name"] for lang in data.get("spoken_languages", [])]) if data.get("spoken_languages") else "N/A"
            return {
                "rating": data.get("vote_average"),
                "vote_count": data.get("vote_count"),
                "release_date": data.get("release_date"),
                "runtime": data.get("runtime"),
                "tagline": data.get("tagline"),
                "overview": data.get("overview"),
                "director": directors,
                "cast": cast_details,
                "genres": genres,
                "budget": budget,
                "revenue": revenue,
                "available_in": available_in,
            }
    except Exception as e:
        print("get_movie_details error:", e)
    return None

# ------------------------------
# Load Data
# ------------------------------
# Expect a DataFrame with at least: ['imdb_id', 'original_title', ... , 'director']
# and a similarity matrix built in the same row order.
# Try relative paths first; fall back to simple names if needed.
movies = None
similarity = None
load_errors = []

candidates = [
    r"C:\Users\Himanshu\Downloads\Sentiment-Analysis-NLP\notebooks_and_related_files\recommendation\pickle\movie_list.pkl",
]

sim_candidates = [
    r"C:\Users\Himanshu\Downloads\Sentiment-Analysis-NLP\notebooks_and_related_files\recommendation\pickle\similarity.pkl",
]

for p in candidates:
    try:
        movies = pickle.load(open(p, "rb"))
        break
    except Exception as e:
        load_errors.append((p, str(e)))

for p in sim_candidates:
    try:
        similarity = pickle.load(open(p, "rb"))
        break
    except Exception as e:
        load_errors.append((p, str(e)))

if movies is None or similarity is None:
    st.error("Could not load movie_list.pkl or similarity.pkl. Check paths/files.")
    if load_errors:
        with st.expander("Load errors (debug)"):
            for path, err in load_errors:
                st.write(f"{path}: {err}")
    st.stop()

# Ensure expected columns exist / are named right
# We need 'original_title' and 'imdb_id' at minimum.
if "original_title" not in movies.columns:
    st.error("The movies dataframe must contain an 'original_title' column.")
    st.stop()

if "imdb_id" not in movies.columns:
    st.error("The movies dataframe must contain an 'imdb_id' column.")
    st.stop()

# Optional director column; if missing we’ll show N/A
has_director_col = "director" in movies.columns

# Normalize types
movies["original_title"] = movies["original_title"].astype(str)
movies["imdb_id"] = movies["imdb_id"].astype(str)

# ------------------------------
# Core Recommender Helpers
# ------------------------------
def recommend(movie_title: str):
    """
    movie_title: the value from 'original_title'
    Return list of 5 recommendations with poster & trailer (via imdb_id).
    """
    index = movies[movies["original_title"] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommendations = []
    for i in distances[1:6]:  # skip the first (itself), take next 5
        row = movies.iloc[i[0]]
        rec_imdb_id = row.imdb_id
        poster = fetch_poster_by_imdb(rec_imdb_id)
        recommendations.append({
            "title": row.original_title,
            "poster": poster,
            "trailer": fetch_trailer_by_imdb(rec_imdb_id)
        })
    return recommendations

def get_random_movie():
    random_movie = movies.sample(1).iloc[0]
    return {
        "title": random_movie["original_title"],
        "poster": fetch_poster_by_imdb(random_movie["imdb_id"]),
        "trailer": fetch_trailer_by_imdb(random_movie["imdb_id"]),
        "imdb_id": random_movie["imdb_id"]
    }

def update_history(imdb_id):
    # Add the movie to recently viewed if it's not the same as the last viewed
    if not st.session_state.history or st.session_state.history[-1] != imdb_id:
        st.session_state.history.append(imdb_id)
        if len(st.session_state.history) > 5:
            st.session_state.history.pop(0)

def get_trending_movies():
    # This is independent of your dataset; it's just TMDB's weekly trending.
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
        response = requests_retry_session().get(url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            trending = data.get("results", [])[:5]
            trending_list = []
            for movie in trending:
                trending_list.append({
                    "title": movie.get("title"),
                    "poster": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get("poster_path") else None,
                    "tmdb_id": movie.get("id"),
                })
            return trending_list
        else:
            return []
    except Exception as e:
        print("get_trending_movies error:", e)
        return []

# ------------------------------
# UI Configuration and Header
# ------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 0.5em;'>
        Let’s Find the Perfect Movie that Matches Your Vibe!🎬
    </h1>
    <p style='text-align: center; color: #7f8c8d; font-size: 1.5rem; margin-top: 0;'>
        Just pick a title and let us do the magic ✨
    </p>

""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Trending Movies Section
# ------------------------------
st.markdown("""
    <h2 style='text-align: center; color: #FF4B4B; margin-bottom: 1rem;'>
        🔥 Now Trending
    </h2>
""", unsafe_allow_html=True)

trending_movies = get_trending_movies()
trending_cols = st.columns(5)
for idx, movie in enumerate(trending_movies):
    with trending_cols[idx]:
        if movie.get("poster"):
            st.image(movie["poster"], use_container_width=True)
        # Now simply display the movie title (centered) without a button
        st.markdown(f"<p style='text-align:center;'>{movie['title']}</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Main Selection Section
# ------------------------------
col_search, col_spacer, col_surprise = st.columns([3, 1, 2])

with col_search:
    st.subheader("🔍 Search for a Movie")
    # Use original_title from your DF instead of title
    selected_movie = st.selectbox("Type to search...", movies["original_title"].values, key="select_movie", help="Start typing to find your movie")
    if st.button("Show Details & Recommendations", key="show_details"):
        st.session_state.mode = "search"
        st.session_state.selected_movie = selected_movie
        st.balloons()

with col_surprise:
    st.subheader("🎭 Let the Algorithm Decide!")
    if st.button("Surprise Me!", key="surprise_me"):
        st.session_state.mode = "surprise"
        st.session_state.random_movie = get_random_movie()
        st.balloons()

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# Content Section: Movie Details & Recommendations
# ------------------------------
if "mode" in st.session_state and st.session_state.mode:
    if st.session_state.mode == "search":
        movie_title = st.session_state.selected_movie
        movie_row = movies[movies["original_title"] == movie_title].iloc[0]
        imdb_id = movie_row.imdb_id
        director_name = movie_row["director"] if has_director_col and pd.notna(movie_row["director"]) else "N/A"

        update_history(imdb_id)
        details = get_movie_details_by_imdb(imdb_id, director_name=director_name)
        trailer_url = fetch_trailer_by_imdb(imdb_id)

        st.markdown("<div style='border-top: 2px solid #eee; margin: 2rem 0;'></div>", unsafe_allow_html=True)
        # Highlighting the movie name in red using HTML inside the markdown
        st.markdown(f"<h2>🎬 Details for: <span style='color: #FF4B4B;'>{movie_title}</span></h2>", unsafe_allow_html=True)

        # Display poster and details side-by-side
        detail_col_left, detail_col_right = st.columns([1, 2])
        with detail_col_left:
            poster = fetch_poster_by_imdb(imdb_id)
            if poster:
                st.image(poster, use_container_width=True)
        with detail_col_right:
            if details:
                # Group 1: Ratings & Runtime
                st.markdown("#### Ratings & Runtime")
                info_cols = st.columns([1, 1, 1])
                with info_cols[0]:
                    rating = details.get('rating', 'N/A')
                    st.markdown(f"**Rating:** <span style='color:green;'>{rating}</span>/10", unsafe_allow_html=True)
                with info_cols[1]:
                    vote_count = details.get('vote_count', 'N/A')
                    st.markdown(f"**No. of Ratings:** <span style='color:green;'>{vote_count}</span>", unsafe_allow_html=True)
                with info_cols[2]:
                    runtime = f"{details.get('runtime', 'N/A')} mins" if details.get('runtime') else "N/A"
                    st.markdown(f"**Runtime:** <span style='color:green;'>{runtime}</span>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                # Tagline in a blue info box
                if details.get("tagline"):
                    st.info(details["tagline"])
                # Overview
                st.markdown("**Overview:**")
                st.write(details.get("overview", "N/A"))

                st.markdown("<br>", unsafe_allow_html=True)
                # Group 2: Release & Financials
                st.markdown("#### Release & Financials")
                row1_cols = st.columns([1, 1, 1])
                with row1_cols[0]:
                    st.markdown(f"**Release Date:** {details.get('release_date', 'N/A')}")
                with row1_cols[1]:
                    st.markdown(f"**Budget:** {details.get('budget', 'N/A')}")
                with row1_cols[2]:
                    st.markdown(f"**Revenue:** {details.get('revenue', 'N/A')}")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # Group 3: Production Details
                st.markdown("#### Production Details")
                row2_cols = st.columns([1, 1, 1])
                with row2_cols[0]:
                    st.markdown(f"**Genres:** {details.get('genres', 'N/A')}")
                with row2_cols[1]:
                    st.markdown(f"**Available in:** {details.get('available_in', 'N/A')}")
                with row2_cols[2]:
                    st.markdown(f"**Directed by:** {details.get('director', 'N/A')}")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # Cast Section
                if details.get("cast"):
                    st.markdown("#### Cast")
                    cast_cols = st.columns(len(details["cast"]))
                    for idx, actor in enumerate(details["cast"]):
                        with cast_cols[idx]:
                            if actor.get("profile"):
                                st.image(actor["profile"], use_container_width=True)
                            st.caption(f"{actor.get('name')} as {actor.get('character')}")
            else:
                st.error("Could not retrieve movie details. Please try another movie.")

            if trailer_url:
                with st.expander("Watch Trailer"):
                    st.video(trailer_url)

        # Display Recommendations
        with st.spinner("Fetching Recommendations..."):
            recommendations = recommend(movie_title)
        st.markdown("<div style='border-top: 2px solid #eee; margin: 2rem 0;'></div>", unsafe_allow_html=True)
        st.subheader("🚀 Recommended Movies")
        rec_cols = st.columns([1, 1, 1])
        for idx, rec in enumerate(recommendations):
            with rec_cols[idx % 3]:
                if rec["poster"]:
                    st.image(rec["poster"], use_container_width=True)
                st.markdown(f"<p style='text-align:center;'><strong>{rec['title']}</strong></p>", unsafe_allow_html=True)
                if rec.get("trailer"):
                    with st.expander("Trailer"):
                        st.video(rec["trailer"])
                        
    elif st.session_state.mode == "surprise":
        random_data = st.session_state.random_movie
        movie_title = random_data["title"]
        imdb_id = random_data.get("imdb_id")
        if not imdb_id:
            movie_row = movies[movies["original_title"] == movie_title].iloc[0]
            imdb_id = movie_row.imdb_id
        director_name = "N/A"
        if has_director_col:
            try:
                director_name = movies[movies["original_title"] == movie_title].iloc[0]["director"]
            except Exception:
                director_name = "N/A"

        update_history(imdb_id)
        details = get_movie_details_by_imdb(imdb_id, director_name=director_name)
        trailer_url = fetch_trailer_by_imdb(imdb_id)

        st.markdown("<div style='border-top: 2px solid #eee; margin: 2rem 0;'></div>", unsafe_allow_html=True)
        # Highlighting the movie name in red using HTML inside the markdown
        st.markdown(f"<h2>🎉 Your Surprise Movie: <span style='color: #FF4B4B;'>{movie_title}</span></h2>", unsafe_allow_html=True)

        detail_col_left, detail_col_right = st.columns([1, 2])
        with detail_col_left:
            poster = fetch_poster_by_imdb(imdb_id)
            if poster:
                st.image(poster, use_container_width=True)
        with detail_col_right:
            if details:
                # Group 1: Ratings & Runtime
                st.markdown("#### Ratings & Runtime")
                info_cols = st.columns([1, 1, 1])
                with info_cols[0]:
                    rating = details.get('rating', 'N/A')
                    st.markdown(f"**Rating:** <span style='color:green;'>{rating}</span>/10", unsafe_allow_html=True)
                with info_cols[1]:
                    vote_count = details.get('vote_count', 'N/A')
                    st.markdown(f"**No. of Ratings:** <span style='color:green;'>{vote_count}</span>", unsafe_allow_html=True)
                with info_cols[2]:
                    runtime = f"{details.get('runtime', 'N/A')} mins" if details.get('runtime') else "N/A"
                    st.markdown(f"**Runtime:** <span style='color:green;'>{runtime}</span>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                # Tagline in a blue info box
                if details.get("tagline"):
                    st.info(details["tagline"])
                # Overview
                st.markdown("**Overview:**")
                st.write(details.get("overview", "N/A"))

                st.markdown("<br>", unsafe_allow_html=True)
                # Group 2: Release & Financials
                st.markdown("#### Release & Financials")
                row1_cols = st.columns([1, 1, 1])
                with row1_cols[0]:
                    st.markdown(f"**Release Date:** {details.get('release_date', 'N/A')}")
                with row1_cols[1]:
                    st.markdown(f"**Budget:** {details.get('budget', 'N/A')}")
                with row1_cols[2]:
                    st.markdown(f"**Revenue:** {details.get('revenue', 'N/A')}")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # Group 3: Production Details
                st.markdown("#### Production Details")
                row2_cols = st.columns([1, 1, 1])
                with row2_cols[0]:
                    st.markdown(f"**Genres:** {details.get('genres', 'N/A')}")
                with row2_cols[1]:
                    st.markdown(f"**Available in:** {details.get('available_in', 'N/A')}")
                with row2_cols[2]:
                    st.markdown(f"**Directed by:** {details.get('director', 'N/A')}")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # Cast Section
                if details.get("cast"):
                    st.markdown("#### Cast")
                    cast_cols = st.columns(len(details["cast"]))
                    for idx, actor in enumerate(details["cast"]):
                        with cast_cols[idx]:
                            if actor.get("profile"):
                                st.image(actor["profile"], use_container_width=True)
                            st.caption(f"{actor.get('name')} as {actor.get('character')}")
            else:
                st.error("Could not retrieve movie details. Please try another movie.")

            if trailer_url:
                with st.expander("Watch Trailer"):
                    st.video(trailer_url)

# ------------------------------
# Sidebar: Recently Viewed
# ------------------------------
with st.sidebar:
    st.header("🕒 Recently Viewed")
    if st.session_state.history:
        for i, hist_imdb in enumerate(reversed(st.session_state.history)):
            # Find row by imdb_id
            row = movies[movies["imdb_id"] == hist_imdb].iloc[0]
            hist_title = row["original_title"]
            hist_poster = fetch_poster_by_imdb(hist_imdb)

            history_container = st.container()
            with history_container:
                if hist_poster:
                    st.image(hist_poster, width=100)
                if st.button(
                    hist_title,
                    key=f"hist_{hist_imdb}_{i}",
                    use_container_width=True
                ):
                    st.session_state.mode = "search"
                    st.session_state.selected_movie = hist_title
                    st.session_state.select_movie = hist_title  # Sync selectbox value
                    st.balloons()
                    st.experimental_rerun()
    else:
        st.write("No history yet.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("<div style='border-top: 2px solid #eee; margin: 2rem 0;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #888; padding: 10px; font-size: 1rem;'>
          Made with ♥️ by  <strong>Himanshu</strong>
    </div>
""", unsafe_allow_html=True)
