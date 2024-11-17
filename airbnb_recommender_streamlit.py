import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title='Airbnb New York Listings Recommender', page_icon=':house:', layout='wide', initial_sidebar_state='auto')
st.title('Airbnb New York Listings Recommender')

@st.cache_data
def load_data():
    listings = pd.read_csv('./NewYork/full_merged.csv')
    return listings

listings = load_data()

st.write(listings.head())

@st.cache_resource
def create_tfidf_matrix(listings):
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')
    tfidf_matrix = vectorizer.fit_transform(listings['description'])
    return tfidf_matrix

@st.cache_resource
def calculate_cosine_similarity(_tfidf_matrix):
    cosine_sim = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
    return cosine_sim

@st.cache_resource
def weighted_sum_similarities(description_similarity, image_similarity, polarity_similarity):
    # normalize the values to be in range [0, 1] in case they are not already --- not sure if necessary
    # description_sim = description_similarity / np.max(description_similarity)
    # image_sim = image_similarity / np.max(image_similarity)
    # polarity_sim = polarity_similarity / np.max(polarity_similarity)

    weights = [0.5, 0.3, 0.2] # change weights
    final_similarity = (weights[0] * image_similarity +
                        weights[1] * polarity_similarity +
                        weights[2] * description_similarity)
    return final_similarity

def convert_str_to_array(data, remove_newline=False):
    cleaned_string = data.replace('\n', ' ') if remove_newline else data
    cleaned_string = cleaned_string.strip('[]')
    vector_array = np.fromstring(cleaned_string, sep=' ')
    return vector_array

tfidf_matrix = create_tfidf_matrix(listings)

listings['photo_vector'] = listings['photo_vector'].apply(lambda x: convert_str_to_array(x, remove_newline=True))
img_vectors = np.array(listings['photo_vector'].tolist())

listings['polarity'] = listings['polarity'].apply(lambda x: convert_str_to_array(x))
polarity = np.array(listings['polarity'].tolist())

desc_cosine_sim = calculate_cosine_similarity(tfidf_matrix)
img_cosine_sim = calculate_cosine_similarity(img_vectors)
polarity_cosine_sim = calculate_cosine_similarity(polarity)

final_sim = weighted_sum_similarities(desc_cosine_sim, img_cosine_sim, polarity_cosine_sim)

# def recommend_listings(cosine_sim, listings, idx, top_n):
#     if idx < 0 or idx >= len(cosine_sim):
#         return ["Invalid listing ID!"]
    
#     score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#     top_n_indexes = list(score_series.iloc[1 : top_n + 1].index)
#     recommendations = listings.iloc[top_n_indexes][['id', 'name', 'description', 'picture_url', 'price']]
    
#     return recommendations

def get_recommendations(cosine_sim, listings, listing_index, k=5):
    similar_indices = np.argsort(cosine_sim[listing_index])[::-1][1:k+1]
    recommended_listings = listings.iloc[similar_indices][['id', 'name', 'description', 'picture_url', 'price']]
    return recommended_listings

# 19783
# 8143
# 25

st.sidebar.header('User Input Features')
id = st.sidebar.number_input('Enter the id of the listing', min_value=0, value=0)

st.header('Listing Name')
st.success(listings['name'][id])
st.header('Listing Price')
st.success(listings['price'][id])
st.subheader('Listing Description')
st.write(listings['description'][id])
st.image(listings['picture_url'][id])

if st.sidebar.button('Recommend'):
    if id >= len(listings):
        st.error("Invalid ID! Please enter a valid listing ID.")
    else:
        recommended_listings = get_recommendations(final_sim, listings, id, 10)
        
        if isinstance(recommended_listings, str):
            st.error(recommended_listings)
        else:
            st.header('Recommended Listings')
            
            for _, row in recommended_listings.iterrows():
                st.subheader(row['name'])
                st.success(row['price'])
                st.write(row['description'])
                st.image(row['picture_url'])
