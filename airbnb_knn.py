import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title='Airbnb New York Listings Recommender', page_icon=':house:', layout='wide', initial_sidebar_state='auto')
st.title('Airbnb New York Listings Recommender')

@st.cache_data
def load_data():
    # listings = pd.read_csv('./NewYork/full_merged_translated.csv')
    listings = pd.read_csv('./full_removed_toosimilar.csv')
    return listings

listings = load_data()
data = np.load('combined_features.npz', allow_pickle=False)
combined_features = data['combined_features']

@st.cache_resource
def load_knn_model():
    knn_model = joblib.load('knn_model.pkl')
    return knn_model

knn_model = load_knn_model()

def get_knn_recommendations(knn_model, listing_index, listings, k=10):
    _, indices = knn_model.kneighbors([combined_features[listing_index]])
    recommendations = listings.iloc[indices[0][1:k+1]][['id', 'name', 'description', 'picture_url', 'price']]
    return recommendations

st.sidebar.header('User Input Features')

listing_names = listings['name'].tolist()
selected_listing_name = st.sidebar.selectbox('Select a listing by name', listing_names)

# selected_listing_index = listings[listings['name'] == selected_listing_name].index[0]
selected_listing_index = 25 # 25, 19853, 20253

st.header('Listing Name')
st.success(listings['name'][selected_listing_index])
st.header('Listing Price')
st.success(listings['price'][selected_listing_index])
st.subheader('Listing Description')
st.write(listings['description'][selected_listing_index])
st.image(listings['picture_url'][selected_listing_index])

if st.sidebar.button('Recommend'):
    recommended_listings = get_knn_recommendations(knn_model, selected_listing_index, listings, k=10)

    st.header('Recommended Listings')
    for _, row in recommended_listings.iterrows():
        st.subheader(row['name'])
        st.success(row['price'])
        st.write(row['description'])
        st.image(row['picture_url'])
