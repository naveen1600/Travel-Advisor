#  {'boolean': ['ByAppointmentOnly', 'Open24Hours'], 'dropdown': {'BusinessParking': 'garage', 'Ambience': 'trendy', 'cuisine': 'cambodian'}}

def get_restaurant_recommendations(user_state, user_city, restaurant_facilities):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt 
    import pandas as pd


    # Load the dataset
    file_path = "flattened_restaurant_details.json"
    data = pd.read_json(file_path)

    data['city_state'] = data['city'] + ', ' + data['state']

    weights = {
    "Parking": 1,
    "Accepts Credit Cards": 1,
    "Ambience": 2,
    "Dogs Allowed": 1,
    "Open 24 Hours": 1,
    "Appointment Only": 1,
    "Good For Kids": 1,
    "Alcohol": 1,
    "Price": 1,
    "Wheel-chair Accessible": 1,
    "Drive Thru": 1,
    "Cuisine": 3,
    "city_state" : 5
    }

    attributes = [
        "Parking", "Accepts Credit Cards", "Ambience", "Dogs Allowed",
        "Open 24 Hours", "Appointment Only", "Good For Kids", "Alcohol",
        "Price", "Wheel-chair Accessible", "Drive Thru", "Cuisine", "city_state"
    ]

    # Filter the dataset to include only the required attributes
    data = data[attributes]

    # Handle missing values for list columns (convert None or NaN to empty lists)
    list_columns = ["Parking", "Ambience", "Cuisine"]
    for col in list_columns:
        data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])

    # MultiLabelBinarizer for element-wise encoding of list columns
    mlb = MultiLabelBinarizer()
    encoded_dfs = []  # To store the encoded DataFrames

    for col in list_columns:
        # Apply MultiLabelBinarizer to encode each list-based column
        encoded = pd.DataFrame(mlb.fit_transform(data[col]),
                            columns=[f"{col}_{label}" for label in mlb.classes_],
                            index=data.index)  # Keep the same index for merging
        encoded_dfs.append(encoded)  # Store the encoded DataFrame

    # Concatenate all encoded list columns
    encoded_data = pd.concat(encoded_dfs, axis=1)

    # One-hot encode 'city_state' column
    city_state_encoded = pd.get_dummies(data['city_state'], prefix='city_state').astype(int)

    # Ensure boolean columns are converted to integers (0 or 1)
    boolean_columns = [
        "Accepts Credit Cards", "Dogs Allowed", "Open 24 Hours", "Appointment Only",
        "Good For Kids", "Alcohol", "Wheel-chair Accessible", "Drive Thru"
    ]
    data[boolean_columns] = data[boolean_columns].astype(int)

    # Extract and include the numeric column 'price'
    numeric_column = ["Price"]

    # Apply weights to the encoded features
    # Multiply each encoded feature by the corresponding weight
    for col in encoded_data:
        encoded_data[col] *= weights.get(col, 1)  # Use weight from the dictionary, default to 1 if not found
    
    # Apply weights to the boolean columns
    for col in boolean_columns:
        data[col] *= weights.get(col, 1)  # Use weight from the dictionary, default to 1 if not found
    
    # Apply weights to the numeric column 'price'
    data[numeric_column] *= weights.get("Price", 1)
    
    # Apply weight to 'city_state' one-hot encoding
    city_state_encoded *= weights.get("city_state", 1)

    # Combine encoded list columns, one-hot encoded city_state, boolean columns, and numeric column into the final dataset
    final_data = pd.concat([encoded_data, city_state_encoded, data[boolean_columns + numeric_column]], axis=1)

    # Display all columns in the DataFrame
    pd.set_option('display.max_columns', None)

    # KNN
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(final_data)

    user_defaults = {
    "Parking": ["validated", "garage", "lot", "valet", "street"],
    "Accepts Credit Cards": False,
    "Ambience": ["classy", "romantic", "upscale", "hipster", "trendy", "casual", "touristy", "intimate", "divey"],
    "Dogs Allowed": False,
    "Open 24 Hours": False,
    "Appointment Only": False,
    "Good For Kids": False,
    "Alcohol": False,
    "Price": 2,
    "Wheel-chair Accessible": False,
    "Drive Thru": False,
    "Cuisine": ['middle eastern', 'hawaiian', 'french', 'chinese', 'filipino', 'american', 'brazilian', 'japanese curry', 'portuguese', 'afghan', 'burmese', 'singaporean', 'new mexican cuisine', 'moroccan', 'cambodian', 'mexican', 'ethiopian', 'indonesian', 'spanish', 'turkish', 'armenian', 'caribbean', 'german', 'ukrainian', 'japanese', 'austrian', 'polish', 'australian', 'syrian', 'cuban', 'bangladeshi', 'peruvian', 'mediterranean', 'egyptian', 'georgian', 'korean', 'italian', 'latin american', 'indian', 'scottish', 'cajun/creole', 'irish', 'himalayan/nepalese', 'hungarian', 'traditional chinese medicine', 'soul food', 'greek', 'mongolian', 'russian', 'vietnamese', 'thai', 'irish pub', 'african', 'south african', 'lebanese', 'malaysian', 'pakistani', 'israeli', 'belgian', 'cantonese', 'uzbek'],
    "city_state" : ""
    }

    def fill_missing_values(user_input, defaults):
        filled_input = {}
        for key, default in defaults.items():
            filled_input[key] = user_input.get(key, default)
        
        # Combine city and state to form city_state
        city = user_input.get("city", defaults.get("city", "Unknown"))  # Use user_input or defaults
        state = user_input.get("state", defaults.get("state", "Unknown"))
        filled_input["city_state"] = f"{city}, {state}"  # Always compute city_state
        
        # Remove city and state from the filled input
        filled_input.pop("city", None)
        filled_input.pop("state", None)
        
        return filled_input

    # Example user input (partial or empty)
    user_input = {
        "city": user_city,
        "state": user_state
    }

    # Fill missing values and remove city/state
    user_input_filled = fill_missing_values(user_input, user_defaults)
    
    if 'boolean' in restaurant_facilities:
        for facility in restaurant_facilities['boolean']:
            user_input_filled[facility] = True

    if 'dropdown' in restaurant_facilities:
        for key, value in restaurant_facilities['dropdown'].items():
            if key in user_defaults:
                user_input_filled[key] = [value] if isinstance(user_defaults[key], list) else value


    ################# Encoding User input ###################3
    data = pd.DataFrame([user_input_filled])

    list_columns = ["Parking", "Ambience", "Cuisine"]

    # MultiLabelBinarizer for element-wise encoding of list columns
    mlb = MultiLabelBinarizer()
    encoded_dfs = []  # To store the encoded DataFrames

    for col in list_columns:
        # Apply MultiLabelBinarizer to encode each list-based column
        encoded = pd.DataFrame(mlb.fit_transform(data[col]),
                            columns=[f"{col}_{label}" for label in mlb.classes_],
                            index=data.index)  # Keep the same index for merging
        encoded_dfs.append(encoded)  # Store the encoded DataFrame

    # Concatenate all encoded list columns
    user_encoded_data = pd.concat(encoded_dfs, axis=1)

    # One-hot encode 'city_state' column
    user_city_state_encoded = pd.get_dummies(data['city_state'], prefix='city_state').astype(int)

    # Ensure boolean columns are converted to integers (0 or 1)
    boolean_columns = [
        "Accepts Credit Cards", "Dogs Allowed", "Open 24 Hours", "Appointment Only",
        "Good For Kids", "Alcohol", "Wheel-chair Accessible", "Drive Thru"
    ]
    data[boolean_columns] = data[boolean_columns].astype(int)

    # Extract and include the numeric column 'price'
    numeric_column = ["Price"]

    # Apply weights to the encoded features
    # Multiply each encoded feature by the corresponding weight
    for col in encoded_data:
        encoded_data[col] *= weights.get(col, 1)  # Use weight from the dictionary, default to 1 if not found
    
    # Apply weights to the boolean columns
    for col in boolean_columns:
        data[col] *= weights.get(col, 1)  # Use weight from the dictionary, default to 1 if not found
    
    # Apply weights to the numeric column 'price'
    data[numeric_column] *= weights.get("Price", 1)
    
    # Apply weight to 'city_state' one-hot encoding
    city_state_encoded *= weights.get("city_state", 1)

    # Combine encoded list columns, one-hot encoded city_state, boolean columns, and numeric column into the final dataset
    final_user_data = pd.concat([user_encoded_data, user_city_state_encoded, data[boolean_columns + numeric_column]], axis=1)

    final_user_data_1 = {col: 0 for col in final_data.columns}

    for col in final_user_data.columns:
        final_user_data_1[col] = final_user_data[col].iloc[0]

    # Convert final_user_data_1 to DataFrame for KNN
    final_user_data_1_df = pd.DataFrame([final_user_data_1])

    distances, indices = knn.kneighbors(final_user_data_1_df)

    flattened_data = pd.read_json('flattened_restaurant_details.json')
    # print(flattened_data.iloc[indices[0]])
    # For multiple rows
    result = flattened_data.iloc[indices[0]].to_dict(orient='records')
    return result



