import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import styles from './ResultsPage.module.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { city, state, startDate, endDate, budget, hotelFacilities, restaurantFacilities } = location.state || {};

  const [tab, setTab] = useState('hotels');
  const [results, setResults] = useState({
    hotels: [],
    restaurants: [],
    attractions: [],
  });

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.post('http://localhost:5000/process-inputs', {
          city,
          state,
          startDate,
          endDate,
          budget,
          hotelFacilities,
          restaurantFacilities,
        });

        // Set the results returned from the backend
        setResults(response.data);
      } catch (error) {
        console.error('Error fetching results:', error);
      }
    };
    fetchResults();
  }, [city, state, startDate, endDate, budget, hotelFacilities, restaurantFacilities]);

  const renderContent = () => {

    if (!results || !results.hotels || !results.restaurants || !results.attractions) {
      return <div>Loading results...</div>; // Show a fallback or loading state
    }

    switch (tab) {
      case 'hotels':
        return (
          <div>
            <h2>Popular Hotels in {city}, {state}</h2>
            {results.hotels.length > 0 ? (
              <ul>
                {results.hotels.map((hotel, index) => (
                  <li key={index}>{hotel}</li>
                ))}
              </ul>
            ) : (
              <p>No hotels found.</p>
            )}
          </div>
        );
      case 'restaurants':
        return (
          <div>
            <h2>Popular Restaurants in {city}, {state}</h2>
            {results.restaurants.length > 0 ? (
              <ul>
                {results.restaurants.map((restaurant, index) => (
                  <li key={index}>{restaurant}</li>
                ))}
              </ul>
            ) : (
              <p>No restaurants found.</p>
            )}
          </div>
        );
      case 'attractions':
        return (
          <div>
            <h2>Popular Attractions in {city}, {state}</h2>
            {results.attractions.length > 0 ? (
              <ul>
                {results.attractions.map((attraction, index) => (
                  <li key={index}>{attraction}</li>
                ))}
              </ul>
            ) : (
              <p>No attractions found.</p>
            )}
          </div>
        );
      default:
        return null;
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
