# from flask import Flask, request, jsonify
# from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping


# app = Flask(__name__)
# CORS(app)

def get_recommendations(state, city, dataset_path='tourist_attractions.csv'):
    # Load your tourist attractions dataset
    tourist_df = pd.read_csv(dataset_path)
    

    # Check for leading/trailing spaces in column names
    tourist_df.columns = tourist_df.columns.str.strip()

    # Ensure 'City' column exists in the dataset
    # if 'City' not in tourist_df.columns:
    #     raise KeyError("The dataset must include a 'City' column.")

    # Create a combined 'State_City' column
    # tourist_df['State_City'] = tourist_df['State'] + ', ' + tourist_df['City']

    # Create a user-item interaction matrix using State and City as combined index (users)
    user_attractions_matrix = tourist_df.pivot_table(
        index='State',  # Use State and City as combined index
        columns='Attraction',  # Attractions as items
        values='Rating',  # Rating as interaction level
        aggfunc='mean'
    )

    # Handle missing values
    user_attractions_matrix = user_attractions_matrix.fillna(0)

    # Normalize the data (min-max scaling) for each state and city individually
    scaler = MinMaxScaler()
    user_attractions_matrix_scaled = scaler.fit_transform(user_attractions_matrix)

    # Create user and item interaction pairs for training
    user_ids = np.array(user_attractions_matrix.index)
    item_ids = np.array(user_attractions_matrix.columns)
    user_indices = np.arange(len(user_ids))
    item_indices = np.arange(len(item_ids))

    user_interactions = []
    item_interactions = []
    ratings = []

    for user in user_indices:
        for item in item_indices:
            rating = user_attractions_matrix_scaled[user, item]
            if rating > 0:  # Only include rated items
                user_interactions.append(user)
                item_interactions.append(item)
                ratings.append(rating)

    user_interactions = np.array(user_interactions)
    item_interactions = np.array(item_interactions)
    ratings = np.array(ratings)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        np.vstack((user_interactions, item_interactions)).T, 
        ratings, 
        test_size=0.2, 
        random_state=42
    )

    # Define the inputs
    X_train_user_input = X_train[:, 0]
    X_train_item_input = X_train[:, 1]
    X_test_user_input = X_test[:, 0]
    X_test_item_input = X_test[:, 1]

    # Create the model with correctly structured inputs
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embedding = Embedding(input_dim=len(user_ids), output_dim=50, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=len(item_ids), output_dim=50, name='item_embedding')(item_input)
    user_vecs = Flatten(name='user_flatten')(user_embedding)
    item_vecs = Flatten(name='item_flatten')(item_embedding)
    y = Dot(axes=1)([user_vecs, item_vecs])
    model = Model(inputs=[user_input, item_input], outputs=y)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with increased complexity and longer patience
    history = model.fit(
        [X_train_user_input, X_train_item_input], 
        y_train, 
        epochs=200,  # Increased epochs
        batch_size=64,  # Adjusted batch size for better learning
        validation_data=([X_test_user_input, X_test_item_input], y_test), 
        callbacks=[early_stopping]
    )

    # Make predictions
    predictions = model.predict([X_test_user_input, X_test_item_input])

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    

    # Define the number of top recommendations
    top_n = 5

    # Check if the state and city combination exists in the data
    state_city_input = f"{state}"
    if state_city_input in user_attractions_matrix.index:
        user_index = np.where(user_ids == state_city_input)[0][0]
        state_city_ratings = user_attractions_matrix_scaled[user_index, :].reshape(1, -1)
        
        # Get predictions for the state and city
        state_city_predictions = model.predict([np.full(len(item_indices), user_index), item_indices])
        
        # Filter predictions to include only relevant attractions from the specified state and city
        relevant_attractions = tourist_df[(tourist_df['State'] == state)]['Attraction']
        relevant_attraction_indices = [item_ids.tolist().index(attraction) for attraction in relevant_attractions if attraction in item_ids]
        state_city_recommendations = [item_ids[idx] for idx in state_city_predictions.flatten().argsort()[::-1] if idx in relevant_attraction_indices][:top_n]
        
        print(state_city_recommendations)
        return state_city_recommendations
    else:
        print(f"State '{state}' and City '{city}' combination not found in the dataset.")
        return []

# @app.route('/recommendations', methods=['POST'])
# def recommendations():
#     req_data = request.get_json()
#     state = req_data.get('state')
#     city = req_data.get('city')
#     print("Starting recommendation generation...")  # Log start of generation
#     result = get_recommendations(state, city)
#     print("Final Recommendations:", result)  # Log only the final result
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)












 












