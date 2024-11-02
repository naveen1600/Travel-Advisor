# Travel-Advisor

Travel Advisor 

1. Project Goal:
  The goal of this project is to create a personalized travel plan consisting of detailed recommendations for the duration of a trip including places to stay, attractions and restaurants. This ensures that the user receives a tailor-made itinerary, recommending options for every meal, place to stay, and activities to explore throughout their trip.

2. Importance and Target Audience:
  The importance of this project lies in enhancing the travel experience by providing personalized recommendations. It helps in improving the decision-making process for travelers who are unfamiliar with a location. By providing personalized recommendations, it will help travelers save time and effort in finding places that align with their tastes and preferences. The primary customers would be new travelers, tourists, or anyone planning a trip who wants curated suggestions for hotels, restaurants, and attractions.

3. Data Source:
  We will be utilizing the Yelp dataset to gather comprehensive user details, restaurant information, and associated reviews. Additionally, we will be crawling TripAdvisor to collect data on hotels and attractions.
  1. Yelp Dataset: User details, restaurant information, reviews
  2. TripAdvisor: Hotel details, attraction information

4. Algorithms Used for Analysis:
  Three algorithms will be used for different aspects of the travel plan:
  1. Collaborative Filtering will be used to recommend hotels using a Matrix Factorization method with Alternating Least Squares (ALS).
  2. For attractions, we will use Restricted Boltzmann Machine (RBM) to suggest options based on what users like and past data.
  3. A mixed approach combines Collaborative Filtering (with K-Nearest Neighbors) and Content-Based Filtering (using K-Means clustering) to recommend restaurants.

5. Expected Learnings from the Project:
  1. User Preference Analysis: This project will explore how to analyze user behavior and use past interactions to customize recommendations.
  2. Scalability Techniques: The focus will be on designing systems that can handle large datasets efficiently.
  3. Algorithm Selection: The project will identify which recommendation methods work best for specific applications and user needs.
  4. User Experience Design: It aims to find ways to improve user satisfaction by integrating personalized recommendations into user interfaces
