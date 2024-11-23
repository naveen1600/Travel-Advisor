import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
// import axios from 'axios';
import styles from './HomePage.module.css';

// Import the cities.json file
import citiesData from '../data/cities_states.json';

const HomePage = () => {
  const navigate = useNavigate();

  // State variables
  const [state, setState] = useState('');
  const [city, setCity] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [budget, setBudget] = useState('');
  const [statesList, setStatesList] = useState([]);
  const [citiesList, setCitiesList] = useState([]);

  // Effect to populate states and cities based on citiesData
  useEffect(() => {
    // Extract unique states from citiesData
    const states = [...new Set(citiesData.map((item) => item.state))];
    setStatesList(states);
    setCitiesList(citiesData);
  }, []);

  // Handle when a state is selected to filter cities
  const handleStateChange = (e) => {
    const selectedState = e.target.value;
    setState(selectedState);

    // Filter cities based on selected state
    const filteredCities = citiesData.filter((cityObj) => cityObj.state === selectedState);
    setCitiesList(filteredCities);
    setCity(''); // Reset city selection when state changes
  };

  const handleNext = async () => {
    const userInputs = {
        city: city,
        state: state,
        startDate: startDate?.toISOString(),
        endDate: endDate?.toISOString(),
        budget: budget,
    };

    // try {
    //     const response = await axios.post('http://localhost:5000/process-inputs', userInputs);
    //     const { results } = response.data;

    //     // Navigate to the FilterPage or ResultsPage with the results
    //     navigate('/filter', { state: { city, state, results } });
    // } catch (error) {
    //     console.error('Error fetching results from the server:', error);
    // }

    navigate('/filter', { state: userInputs });
};
  

  // Handle Logout
  const handleLogout = () => {
    navigate('/');
  };

  return (
    <div className={styles.homeContainer}>
      <nav className={styles.navbar}>
        <h1 className={styles.navbarTitle}>Travel Advisor</h1>
        <button className={styles.logoutButton} onClick={handleLogout}>Logout</button>
      </nav>

      <div className={styles.formContainer}>
        {/* Dropdown for State */}
        <label>
          Select State:
          <select value={state} onChange={handleStateChange}>
            <option value="">Select</option>
            {statesList.map((stateName, index) => (
              <option key={index} value={stateName}>{stateName}</option>
            ))}
          </select>
        </label>

        {/* Dropdown for City */}
        <label>
          Select City:
          <select value={city} onChange={(e) => setCity(e.target.value)}>
            <option value="">Select</option>
            {citiesList.map((cityObj, index) => (
              <option key={index} value={cityObj.city}>{cityObj.city}</option>
            ))}
          </select>
        </label>

        <label>
          Travel Start Date:
          <DatePicker className={styles.date} selected={startDate} onChange={(date) => setStartDate(date)} />
        </label>
        <label>
          Travel End Date:
          <DatePicker className={styles.date} selected={endDate} onChange={(date) => setEndDate(date)} />
        </label>
        <label>
          Budget:
          <input
            className={styles.budget}
            type="number"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
          />
        </label>
        <button class = {styles.nextButton} onClick={handleNext}>Next</button>
      </div>
    </div>
  );
};

export default HomePage;
