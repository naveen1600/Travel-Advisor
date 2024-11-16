import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styles from './ResultsPage.module.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { city, state } = location.state || {};

  const [tab, setTab] = useState('hotels');

  const renderContent = () => {
    switch (tab) {
      case 'hotels':
        return <div>Popular hotels in {city}, {state}</div>;
      case 'restaurants':
        return <div>Popular restaurants in {city}, {state}</div>;
      case 'attractions':
        return <div>Popular attractions in {city}, {state}</div>;
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
