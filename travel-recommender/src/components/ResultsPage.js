// ResultsPage.js
import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';

const ResultsPage = () => {
  const location = useLocation();
  const { location: userLocation, startDate, endDate, budget } = location.state || {};

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

  return (
    <div>
      <h1>Results for {userLocation}</h1>
      <div>
        <button onClick={() => setTab('hotels')}>Hotels</button>
        <button onClick={() => setTab('restaurants')}>Restaurants</button>
        <button onClick={() => setTab('attractions')}>Attractions</button>
      </div>
      <div>{renderContent()}</div>
    </div>
  );
};

export default ResultsPage;
