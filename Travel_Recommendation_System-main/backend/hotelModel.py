# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt

# # Load the dataset
# df = pd.read_csv("hotelDataset.csv")

# # Define facility columns
# facilities_columns = [
#     'Spa', 'Airport shuttle', 'Business Center', 'Pool', 'Contactless check', 
#     'Pet friendly', 'Free Wifi', 'Wheel-chair accessible', 'Complementary breakfast', 
#     'Fitness Center', '24 hr front desk', 'Valet Parking', 'Open bar', 'smoking'
# ]
# facilities_data = df[facilities_columns]

# # Step 1: Standardize the facilities data
# scaler = StandardScaler()
# facilities_data_scaled = scaler.fit_transform(facilities_data)

# # Step 2: Apply K-Means clustering
# optimal_k = 7  # Set this based on the elbow method or a chosen value
# kmeans = KMeans(n_clusters=optimal_k, random_state=0)
# df['Cluster'] = kmeans.fit_predict(facilities_data_scaled)

# # Function for hotel recommendation
# def hybrid_recommendation(state, city, user_facilities):
#     # Convert user facilities into a binary preference vector
#     user_preferences = np.array([1 if facility in user_facilities else 0 for facility in facilities_columns])
    
#     # Step 4: Filter hotels by state and city
#     location_filtered_df = df[(df['state'] == state) & (df['city'] == city)]
    
#     # Step 5: Find the best cluster for the user's preferences
#     if not location_filtered_df.empty:
#         # Calculate cosine similarity between user preferences and each cluster centroid
#         cluster_centroids = kmeans.cluster_centers_
#         user_preferences_reshaped = user_preferences.reshape(1, -1)
#         similarities = cosine_similarity(user_preferences_reshaped, cluster_centroids)
#         best_cluster_index = np.argmax(similarities)
        
#         # Step 6: Filter hotels belonging to the best cluster
#         recommended_hotels = location_filtered_df[location_filtered_df['Cluster'] == best_cluster_index]
        
#         # Step 7: Rank hotels by number of facility matches
#         def count_facility_matches(hotel_row, user_preferences):
#             return np.sum(hotel_row.values == user_preferences)
        
#         if not recommended_hotels.empty:
#             recommended_hotels = recommended_hotels.copy()  # Avoid SettingWithCopyWarning
#             recommended_hotels['Facility_Match_Count'] = recommended_hotels[facilities_columns].apply(
#                 lambda row: count_facility_matches(row, user_preferences), axis=1
#             )
            
#             # Step 8: Sort by facility match count, then by ratings if available
#             if 'ratings' in recommended_hotels.columns:
#                 recommended_hotels = recommended_hotels.sort_values(by=['Facility_Match_Count', 'ratings'], ascending=[False, False])
#             else:
#                 recommended_hotels = recommended_hotels.sort_values(by='Facility_Match_Count', ascending=False)
        
#         # Step 9: Fallback if fewer than 3 hotels match
#         if len(recommended_hotels) < 3:
#             additional_hotels_needed = 3 - len(recommended_hotels)
#             fallback_hotels = location_filtered_df[~location_filtered_df.index.isin(recommended_hotels.index)]
            
#             if 'ratings' in fallback_hotels.columns:
#                 fallback_hotels = fallback_hotels.sort_values(by='ratings', ascending=False).head(additional_hotels_needed)
#             else:
#                 fallback_hotels = fallback_hotels.head(additional_hotels_needed)
            
#             recommended_hotels = pd.concat([recommended_hotels, fallback_hotels])

#     else:
#         recommended_hotels = pd.DataFrame()  # Empty dataframe if no hotels found at all
    
#     # Return top recommendations
#     if not recommended_hotels.empty:
#         recommended_hotels_hi=recommended_hotels[['HotelName', 'HotelWebsite', 'HotelRating', 'Address']].head(3)
#         return  recommended_hotels_hi.to_dict(orient='records')
    

# # # Example usage:
# # state = "California"
# # city = "San Francisco"
# # user_facilities = ["Spa", "Free Wifi", "Pool"]

# # recommended_hotels = recommendation(state, city, user_facilities)
# # print(recommended_hotels)

##################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("hotelDataset.csv")
df.head()

# Define facility columns
facilities_columns = [
    'Spa', 'Airport shuttle', 'Business Center', 'Pool', 'Contactless check', 
    'Pet friendly', 'Free Wifi', 'Wheel-chair accessible', 'Complementary breakfast', 
    'Fitness Center', '24 hr front desk', 'Valet Parking', 'Open bar', 'smoking'
]
facilities_data = df[facilities_columns]

# Standardize the facilities data
scaler = StandardScaler()
facilities_data_scaled = scaler.fit_transform(facilities_data)

# Apply K-Means clustering
optimal_k = 7  # Set this based on the elbow method or a chosen value
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df['Cluster'] = kmeans.fit_predict(facilities_data_scaled)

# Define the function for getting hotel recommendations
def hybrid_recommendation(state, city, desired_facilities):
    """
    This function returns the top hotel recommendations based on user input.
    """
    # Convert to binary user preference vector
    print("Desired Fac", desired_facilities)
    user_preferences = np.array([1 if facility.strip() in desired_facilities else 0 for facility in facilities_columns])
    
    # Filter hotels by state and city
    location_filtered_df = df[(df['state'] == state) & (df['city'] == city)]
    
    # Find the best cluster for the user's preferences
    if not location_filtered_df.empty:
        # Calculate cosine similarity between user preferences and each cluster centroid
        cluster_centroids = kmeans.cluster_centers_
        user_preferences_reshaped = user_preferences.reshape(1, -1)
        similarities = cosine_similarity(user_preferences_reshaped, cluster_centroids)
        best_cluster_index = np.argmax(similarities)
        
        # Filter hotels belonging to the best cluster
        recommended_hotels = location_filtered_df[location_filtered_df['Cluster'] == best_cluster_index]
        
        # Rank hotels by number of facility matches
        def count_facility_matches(hotel_row, user_preferences):
            return np.sum(hotel_row.values == user_preferences)
        
        if not recommended_hotels.empty:
            recommended_hotels = recommended_hotels.copy()  # Avoid SettingWithCopyWarning
            recommended_hotels['Facility_Match_Count'] = recommended_hotels[facilities_columns].apply(
                lambda row: count_facility_matches(row, user_preferences), axis=1
            )
            
            # Sort by facility match count, then by ratings if available
            if 'ratings' in recommended_hotels.columns:
                recommended_hotels = recommended_hotels.sort_values(by=['Facility_Match_Count', 'ratings'], ascending=[False, False])
            else:
                recommended_hotels = recommended_hotels.sort_values(by='Facility_Match_Count', ascending=False)
        
        # Fallback if fewer than 3 hotels match
        if len(recommended_hotels) < 3:
            print("\nOnly a few hotels match the exact facilities. Adding top-rated hotels to reach 3 recommendations.")
            additional_hotels_needed = 3 - len(recommended_hotels)
            fallback_hotels = location_filtered_df[~location_filtered_df.index.isin(recommended_hotels.index)]
            
            if 'ratings' in fallback_hotels.columns:
                fallback_hotels = fallback_hotels.sort_values(by='ratings', ascending=False).head(additional_hotels_needed)
            else:
                fallback_hotels = fallback_hotels.head(additional_hotels_needed)
            
            recommended_hotels = pd.concat([recommended_hotels, fallback_hotels])
    
    else:
        print("No hotels found for the specified location.")
        recommended_hotels = pd.DataFrame()  # Empty dataframe if no hotels found at all
    
    # Return the top 3 hotel recommendations as a dictionary
    if not recommended_hotels.empty:
        top_hotels = recommended_hotels[['HotelName', 'HotelWebsite', 'HotelRating', 'Address']].head(3)
        return top_hotels.to_dict(orient='records')
    else:
        return []



# # Example usage:
# state = "IL"
# city = "Fairview Heights"
# desired_facilities = ["Spa", "Free Wifi"]

# recommendations = get_hotel_recommendations(state, city, desired_facilities)
# print(recommendations)

