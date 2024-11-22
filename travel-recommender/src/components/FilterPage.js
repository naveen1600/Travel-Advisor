import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import styles from './FilterPage.module.css';

const FilterPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { city, state, startDate, endDate, budget } = location.state || {};
  const [selectedHotelFacilities, setSelectedHotelFacilities] = useState([]);
  const [selectedRestaurantFacilities, setSelectedRestaurantFacilities] = useState([]);

  const hotelFacilities = ["Free WiFi", "Parking", "Swimming Pool", "Gym", "Spa"];
  const restaurantFacilities = ["Outdoor Seating", "Pet Friendly", "Free WiFi", "Live Music", "Family Friendly"];

  const handleFilterSearch = async () => {
    try {
      const response = await axios.post('http://localhost:5000/process-inputs', {
        city,
        state,
        startDate,
        endDate,
        budget,
        hotelFacilities: selectedHotelFacilities,
        restaurantFacilities: selectedRestaurantFacilities,
      });

      const results = response.data; // Assume this contains the ML results
      navigate('/results', { state: { results, city, state } });
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const handleFacilityChange = (facility, isChecked, type) => {
    if (type === 'hotel') {
      setSelectedHotelFacilities((prev) =>
        isChecked ? [...prev, facility] : prev.filter((item) => item !== facility)
      );
    } else {
      setSelectedRestaurantFacilities((prev) =>
        isChecked ? [...prev, facility] : prev.filter((item) => item !== facility)
      );
    }
  };

  const handleGoBack = () => {
    navigate('/home'); // Redirects to HomePage
  };

  return (
    <div className={styles.filterPage}>
      <nav className={styles.navbar}>
        <h1 className={styles.navbarTitle}>Travel Advisor</h1>
      </nav>
      <div className={styles.filterContainer}>
        {/* Left section for Restaurant Facilities */}
        <div className={styles.filterLeft}>
          <h2>Restaurant Facilities</h2>
          <div className={styles.checkboxGroup}>
            {restaurantFacilities.map((facility, index) => (
              <div className={styles.checkboxItem} key={index}>
                <label>{facility}</label>
                <input 
                  type="checkbox" 
                  id={`restaurant-${index}`} 
                  value={facility}
                  onChange={(e) => handleFacilityChange(facility, e.target.checked, 'restaurant')}
                />
              </div>
            ))}
          </div>
        </div>
  
        {/* Right section for Hotel Facilities */}
        <div className={styles.filterRight}>
          <h2>Hotel Facilities</h2>
          <div className={styles.checkboxGroup}>
            {hotelFacilities.map((facility, index) => (
              <div className={styles.checkboxItem} key={index}>
                <label>{facility}</label>
                <input 
                  type="checkbox" 
                  id={`hotel-${index}`} 
                  value={facility}
                  onChange={(e) => handleFacilityChange(facility, e.target.checked, 'hotel')}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
  
      {/* Buttons at the bottom */}
      <div className={styles.buttonGroup}>
        <button onClick={handleGoBack} className={styles.prevButton}>
          Previous
        </button>
        <button onClick={handleFilterSearch} className={styles.searchButton}>
          Search
        </button>
      </div>
    </div>
  );  
};

export default FilterPage;
