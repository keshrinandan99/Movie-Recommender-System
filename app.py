import pickle
import streamlit as st
import requests
import pandas as pd

# Configure page first (must be first Streamlit command)
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Initialize session state
if 'movies' not in st.session_state:
    st.session_state.movies = None
if 'similarity' not in st.session_state:
    st.session_state.similarity = None

@st.cache_data
def load_data():
    """Load preprocessed data with caching"""
    try:
        movies = pickle.load(open('movie_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except FileNotFoundError:
        st.error("❌ Required data files not found! Please ensure 'movie_list.pkl' and 'similarity.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API with better error handling"""
    if pd.isna(movie_id) or movie_id == 0:
        return "https://via.placeholder.com/500x750/cccccc/666666?text=No+Poster"
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        
        # Return placeholder if no poster found
        return "https://via.placeholder.com/500x750/cccccc/666666?text=No+Poster"
        
    except requests.exceptions.Timeout:
        return "https://via.placeholder.com/500x750/ffcccc/cc0000?text=Timeout"
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750/ffcccc/cc0000?text=Error"
    except Exception:
        return "https://via.placeholder.com/500x750/cccccc/666666?text=No+Image"

def recommend(movie, movies_df, similarity_matrix):
    """Get movie recommendations based on similarity"""
    try:
        # Check if movie exists
        movie_matches = movies_df[movies_df['title'] == movie]
        if movie_matches.empty:
            return [], []
        
        # Get the index of the selected movie
        index = movie_matches.index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        recommended_movie_names = []
        recommended_movie_posters = []
        
        # Get top 5 similar movies (excluding the selected movie itself)
        for i in distances[1:6]:
            movie_title = movies_df.iloc[i[0]]['title']
            movie_id = movies_df.iloc[i[0]]['id']
            
            recommended_movie_names.append(movie_title)
            poster_url = fetch_poster(movie_id)
            recommended_movie_posters.append(poster_url)

        return recommended_movie_names, recommended_movie_posters
    
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return [], []

# Main App
def main():
    st.title('🎬 Movie Recommender System')
    st.markdown("---")
    
    # Load data
    with st.spinner('Loading movie data...'):
        movies, similarity = load_data()
    
    if movies is None or similarity is None:
        st.error("Failed to load required data files.")
        return
    
    # Check data structure
    if 'title' not in movies.columns:
        st.error("❌ Invalid data format: 'title' column not found in movies data.")
        return
    
    if 'id' not in movies.columns:
        st.error("❌ Invalid data format: 'id' column not found in movies data.")
        return
    
    # Create movie selection dropdown
    movie_list = sorted(movies['title'].dropna().unique())
    
    if len(movie_list) == 0:
        st.error("❌ No movies found in the dataset.")
        return
    
    selected_movie = st.selectbox(
        "🔍 Type or select a movie from the dropdown:",
        options=movie_list,
        help="Choose a movie to get similar recommendations"
    )
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recommendation button
    if st.button('🎯 Show Recommendations', type="primary"):
        if selected_movie:
            with st.spinner('Finding similar movies...'):
                recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity)
            
            if recommended_movie_names and len(recommended_movie_names) > 0:
                st.success(f"Here are movies similar to **{selected_movie}**:")
                st.markdown("---")
                
                # Display recommendations in columns
                cols = st.columns(5)
                
                for i, (name, poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
                    if i < 5:  # Ensure we don't exceed 5 columns
                        with cols[i]:
                            st.markdown(f"**{name}**")
                            try:
                                st.image(poster, use_container_width=True)
                            except Exception as e:
                                st.error(f"Failed to load image: {e}")
                                st.image("https://via.placeholder.com/500x750/cccccc/666666?text=Error", use_container_width=True)
            else:
                st.warning("⚠️ No recommendations found. Please try a different movie.")
        else:
            st.warning("⚠️ Please select a movie first!")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Movie data from TMDB • Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Add sidebar with info
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown(
            """
            This movie recommender uses:
            - **Content-based filtering**
            - **Cosine similarity**
            - **TF-IDF vectorization**
            
            The system analyzes movie features like:
            - Genres
            - Keywords  
            - Cast
            - Director
            - Plot overview
            """
        )
        
        st.markdown("## 📊 Stats")
        try:
            st.metric("Total Movies", len(movies))
            st.metric("Features Used", "5000+")
        except:
            st.metric("Total Movies", "Loading...")
            st.metric("Features Used", "5000+")
        
        st.markdown("## 🔧 How it works")
        st.markdown(
            """
            1. Select a movie you like
            2. System finds similar movies
            3. Get 5 personalized recommendations
            """
        )
        
        # Debug info (optional - remove in production)
        if st.checkbox("Show Debug Info"):
            st.markdown("### Debug Information")
            if movies is not None:
                st.write(f"Data columns: {list(movies.columns)}")
                st.write(f"Data shape: {movies.shape}")
                st.write(f"Sample movie IDs: {movies['id'].head().tolist()}")

if __name__ == "__main__":
    main()