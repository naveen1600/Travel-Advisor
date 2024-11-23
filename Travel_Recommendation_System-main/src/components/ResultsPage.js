
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import styles from './ResultsPage.module.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { city, state, hotelFacilities, restaurantFacilities } = location.state || {};

  const [tab, setTab] = useState('hotels');
  const [results, setResults] = useState({
    hotels: [],
    restaurants: [],
    attractions: [],
  });

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.post('http://localhost:5001/recommendations', {
          city,
          state,
          hotelFacilities,
          restaurantFacilities
        });
        // console.log("Hotel recommendations:", response.data);

        // Set the results returned from the backend (only hotels in this case)
        setResults({
          hotels: response.data.hotels || [],
          restaurants: response.data.restaurants || [],
          attractions: response.data.attractions || [],
        });
      } catch (error) {
        console.error('Error fetching results:', error);
      }
    };
    fetchResults();
  }, [city, state, hotelFacilities, restaurantFacilities]);

  const renderContent = () => {
    if (!results || !results.hotels) {
      return <div>Loading results...</div>; 
    }
    else {
      switch (tab) {
        case 'hotels':
          return (
            <div className={styles.hotelList}>
              <h2 className={styles.hotelHeading}>Popular Hotels</h2>
              {results.hotels.length > 0 ? (
                <div className={styles.hotelCards}>
                  {results.hotels.map((hotel, index) => (
                    <div className={styles.hotelCard} key={index}>
                      <h3 className={styles.hotelName}>{hotel.HotelName}</h3>
                      <div className={styles.hotelDetails}>
                        <p><strong>Rating:</strong> {hotel.HotelRating}</p>
                        <p><strong>Address:</strong> {hotel.Address}</p>
                        <p><strong>Website:</strong> 
                          <a href={hotel.HotelWebsite} target="_blank" rel="noopener noreferrer">
                            {hotel.HotelWebsite}
                          </a>
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p>No hotels found.</p>
              )}
            </div>
          );

        // You can add similar sections for restaurants or attractions if needed
        
        case 'restaurants':
          return (
            <div className={styles.hotelList}>
              <h2 className={styles.hotelHeading}>Popular Restaurants in {city}, {state}</h2>
              {results.restaurants.length > 0 ? (
                <div className={styles.hotelCards}>
                  {results.restaurants.map((restaurant, index) => (
                    <div className={styles.hotelCard} key={index}>
                      <h3 className={styles.hotelName}>{restaurant.name}</h3>
                      <div className={styles.hotelDetails}>
                        <p><strong>Address: </strong> {restaurant.address}, {restaurant.city}, {restaurant.state}</p> 
                        {restaurant.Cuisine && restaurant.Cuisine.length > 0 && (
                          <p><strong>Cuisine: </strong> 
                            {restaurant.Cuisine.map((item, index) => (
                              <span key={index}>{item}{index < restaurant.Cuisine.length - 1 && ', '}</span>
                            ))}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p>No restaurants found.</p>
              )}
            </div>
          );
          case 'attractions':
            return (
              <div className={styles.hotelList}>
                <h2 className={styles.hotelHeading}>Popular Attractions in {city}, {state}</h2>
                {results.attractions.length > 0 ? (
                  <div className={styles.hotelCards}>
                    {results.attractions 
                    .sort((a, b) => b.rating - a.rating) 
                    .slice(0, 5)
                    .map((attraction, index) => (
                      <div className={styles.hotelCard} key={index}>
                        <h3 className={styles.hotelName}>{attraction.Attraction}</h3>
                        <div className={styles.hotelDetails}>
                          <p><strong>Rating:</strong> {attraction.Rating}</p>
                          <p><strong>Address:</strong> {attraction.Address}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                      <p>No attractions found.</p>
                    )}
                  </div>
            );
          default:
            return null;
      }
    }
  };

  const handleModifySearch = () => {
    navigate('/home');
  };

  return (
    <div className={styles.resultsContainer}>
      <div className={styles.navbar}>
        <div className={styles.tabButtons}>
          <button
            onClick={() => setTab('hotels')}
            className={tab === 'hotels' ? styles.active : ''}
          >
            Hotels
          </button>
          <button
            onClick={() => setTab('restaurants')}
            className={tab === 'restaurants' ? styles.active : ''}
          >
            Restaurants
          </button>
          <button
            onClick={() => setTab('attractions')}
            className={tab === 'attractions' ? styles.active : ''}
          >
            Attractions
          </button>
        </div>
        
        {/* Modify Search button */}
        <button onClick={handleModifySearch} className={styles.modifySearchButton}>
          Modify Search
        </button>
      </div>
      <div className={styles.resultsHeading}>
        <h1>Results for {city}, {state}</h1>  {/* Display city and state */}
      </div>
      {/* Content area */}
      <div className={styles.tabContent}>{renderContent()}</div>
    </div>
  );
};

export default ResultsPage;
