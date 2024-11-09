import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import './HomePage.css';

const HomePage = () => {
  const navigate = useNavigate();
  const [location, setLocation] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [budget, setBudget] = useState('');

  const handleSearch = () => {
    navigate('/results', {
      state: { location, startDate, endDate, budget },
    });
  };

  return (
    <div className="home-container">
      <h1>Travel Recommender</h1>
      <label>
        Select Location:
        <select value={location} onChange={(e) => setLocation(e.target.value)}>
          <option value="">Select a location</option>
          <option value="New York">New York</option>
          <option value="Los Angeles">Los Angeles</option>
          <option value="Chicago">Chicago</option>
          {/* Add more cities */}
        </select>
      </label>
      <label>
        Travel Start Date:
        <DatePicker selected={startDate} onChange={(date) => setStartDate(date)} />
      </label>
      <label>
        Travel End Date:
        <DatePicker selected={endDate} onChange={(date) => setEndDate(date)} />
      </label>
      <label>
        Budget:
        <input
          type="number"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
        />
      </label>
      <button onClick={handleSearch}>Search</button>
    </div>
  );
};

export default HomePage;
