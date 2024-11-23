import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle CORS issues between frontend and backend
from hotelModel import hybrid_recommendation  # Import recommendation logic
from restaurantKNN import get_restaurant_recommendations # Import recommendation logic
from rbm import get_recommendations

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/recommendations', methods=['POST'])
def recommend():
    try:
        print(f"Request received at: {datetime.datetime.now()}")
        # Receive data from the frontend (POST request body)
        data = request.json
        # Extract necessary fields from the received data
        city = data.get('city', '')  # Convert city to lowercase to match dataset
        state = data.get('state', '')  # Convert state to lowercase to match dataset
        hotel_facilities = data.get('hotelFacilities', [])
        # print("HOTEL FACILITIES:", hotel_facilities)
        restaurant_facilities = data.get('restaurantFacilities', [])
        
        # Get hotel recommendations based on the user input
        hotel_recommendations = hybrid_recommendation(state, city, hotel_facilities)
        restaurant_recommendations = get_restaurant_recommendations(state, city, restaurant_facilities)
        attraction_recommendations = get_recommendations(state, city)
        recommendations = {
            'hotels': hotel_recommendations,
            'restaurants': restaurant_recommendations,
            'attractions': attraction_recommendations
        }
        # print("Recommendations:", restaurant_recommendations)
        # Return the recommendations as JSON
        return jsonify(recommendations)

    except Exception as e:
        # If there's any error, return a 500 error with the error message
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Make sure Flask runs on port 5001
