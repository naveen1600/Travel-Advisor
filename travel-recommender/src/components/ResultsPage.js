import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './ResultPage.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { location: userLocation } = location.state || {};

  const [tab, setTab] = useState('hotels');

  const renderContent = () => {
    switch (tab) {
      case 'hotels':
        return <div>List of recommended hotels in {userLocation}</div>;
      case 'restaurants':
        return <div>List of recommended restaurants in {userLocation}</div>;
      case 'attractions':
        return <div>List of recommended attractions in {userLocation}</div>;
      default:
        return null;
    }
  };

  const handleModifySearch = () => {
    navigate('/');
  };

  return (
    <div className="results-container">
      
      {/* Navigation bar */}
      <div className="nav-bar">
        <div className="tab-buttons">
          <button
            onClick={() => setTab('hotels')}
            className={tab === 'hotels' ? 'active' : ''}
          >
            Hotels
          </button>
          <button
            onClick={() => setTab('restaurants')}
            className={tab === 'restaurants' ? 'active' : ''}
          >
            Restaurants
          </button>
          <button
            onClick={() => setTab('attractions')}
            className={tab === 'attractions' ? 'active' : ''}
          >
            Attractions
          </button>
        </div>
        
        {/* Modify Search button */}
        <button onClick={handleModifySearch} className="modify-search-button">
          Modify Search
        </button>
      </div>
      <h1>Results for {userLocation}</h1>
      {/* Content area */}
      <div className="tab-content">{renderContent()}</div>
    </div>
  );
};

export default ResultsPage;
