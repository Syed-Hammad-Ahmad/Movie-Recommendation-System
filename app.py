import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
<style>

.stApp {
    background-color: #141414;
}

h1 {
    color: #00E5FF !important;
    text-align: center;
}

h2, h3 {
    color: white !important;
}

p, label {
    color: white !important;
}

.stButton>button {
    background-color: #00E5FF;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #B20710;
    color: white;
}

.stSelectbox label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<center><h4 style='color:white;'>Discover movies you'll love</h4></center>",
    unsafe_allow_html=True
)

movies = pickle.load(open("movies.pkl","rb"))
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=318679c222cb054bfdc67a9e4ee87ef6"
    data = requests.get(url)
    data = data.json()

    poster_path = data['poster_path']

    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title']==movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x : x[1])

    recommendations = []
    recommendations_posters = []
    for i in movies_list[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommendations.append(movies.iloc[i[0]].title)
        recommendations_posters.append(fetch_poster(movie_id))

    return recommendations, recommendations_posters

#st.title("🎬 Movie Recommendation System")
#st.markdown("Get movie recommendations based on content similarity.")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations, posters = recommend(selected_movie)
    st.subheader("Recommended Movies")
    #for i in range(len(recommendations)):
    #    st.image(posters[i])
    #    st.write(recommendations[i])
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(posters[0])
        st.write(recommendations[0])
    with col2:
        st.image(posters[1])
        st.write(recommendations[1])
    with col3:
        st.image(posters[2])
        st.write(recommendations[2])
    with col4:
        st.image(posters[3])
        st.write(recommendations[3])
    with col5:
        st.image(posters[4])
        st.write(recommendations[4])