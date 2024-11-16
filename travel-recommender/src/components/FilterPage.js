import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './FilterPage.module.css';

const FilterPage = () => {
  const navigate = useNavigate();

  // Lists of facilities for hotels and restaurants
  const hotelFacilities = ["Free WiFi", "Parking", "Swimming Pool", "Gym", "Spa"];
  const restaurantFacilities = ["Outdoor Seating", "Pet Friendly", "Free WiFi", "Live Music", "Family Friendly"];

  const handleFilterSearch = () => {
    navigate('/results'); // Redirects to ResultsPage
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
                <input type="checkbox" id={`restaurant-${index}`} value={facility} />
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
                <input type="checkbox" id={`hotel-${index}`} value={facility} />
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
