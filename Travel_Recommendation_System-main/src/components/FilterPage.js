import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import styles from './FilterPage.module.css';

const FilterPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { city, state, startDate, endDate, budget } = location.state || {};
  const [selectedHotelFacilities, setSelectedHotelFacilities] = useState([]);
  const [selectedRestaurantFacilities, setSelectedRestaurantFacilities] = useState({
    booleanFacilities: [],
    dropdownFacilities: {},
  });

  const hotelFacilities = ["Spa","Airport shuttle","Business Center","Contactless check","Pool","Pet friendly","Free Wifi","Wheel-chair accessible","Complementary breakfast","Fitness Center","24 hr front desk","Valet Parking","Open bar","smoking"];
  const restaurantFacilities = {
    booleanFacilities: [
      "Accepts Credit Cards", "Dogs Allowed", "Open 24 Hours", "Appointment Only", "Good For Kids", "Alcohol", 
      "Wheel-chair Accessible", "Drive Thru",
    ],
    dropdownFacilities: {
      Parking: ["validated", "garage", "lot", "valet", "street"],
      Ambience: ["classy", "romantic", "upscale", "hipster", "trendy", "casual", "touristy", "intimate", "divey"],
      Cuisine: ["middle eastern", "hawaiian", "french", "chinese", "filipino", "american", "brazilian", "japanese curry",
        "portuguese", "afghan", "burmese", "singaporean", "new mexican cuisine", "moroccan", "cambodian", "mexican",
        "ethiopian", "indonesian", "spanish", "turkish", "armenian", "caribbean", "german", "ukrainian", "japanese", "austrian",
        "polish", "australian", "syrian", "cuban", "bangladeshi", "peruvian", "mediterranean", "egyptian", "georgian", "korean",
        "italian", "latin american", "indian", "scottish", "cajun/creole", "irish", "himalayan/nepalese", "hungarian",
        "traditional chinese medicine", "soul food", "greek", "mongolian", "russian", "vietnamese", "thai", "irish pub",
        "african", "south african", "lebanese", "malaysian", "pakistani", "israeli", "belgian", "cantonese", "uzbek"]
    }
  };

  const handleFilterSearch = async () => {
    const payload = {
      city,
      state,
      startDate,
      endDate,
      budget,
      hotelFacilities: selectedHotelFacilities,
      restaurantFacilities: {
        boolean: selectedRestaurantFacilities.booleanFacilities,
        dropdown: selectedRestaurantFacilities.dropdownFacilities,
      },
    };
    navigate('/results', { state: payload });
    // try {
    //   const response = await axios.post('http://localhost:5000/process-inputs', payload);
    //   const results = response.data; // Assume this contains the ML results
    //   navigate('/results', { state: { results, city, state } });
    // } catch (error) {
    //   console.error('Error fetching results:', error);
    // }
  };

  const handleFacilityChange = (facility, value, type, isDropdown = false) => {
    if (type === 'hotel') {
      setSelectedHotelFacilities((prev) => 
        value
            ? [...prev, facility]
            : prev.filter((item) => item !== facility)
      );
    } else if (type === 'restaurant') {
      setSelectedRestaurantFacilities((prev) => {
        if (isDropdown) {
          return {
            ...prev,
            dropdownFacilities: {
              ...prev.dropdownFacilities,
              [facility]: value,
            },
          }
        } else {
          return {
            ...prev,
            booleanFacilities: value
              ? [...prev.booleanFacilities, facility]
              : prev.booleanFacilities.filter((item) => item !== facility)
          };
        }
      });
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
            {restaurantFacilities.booleanFacilities.map((facility, index) => (
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
            {Object.entries(restaurantFacilities.dropdownFacilities).map(([facility, options], index) => (
                <div className={styles.dropdownItem} key={`dropdown-${index}`}>
                  <label>{facility}</label>
                  <select
                    id={`restaurant-dropdown-${index}`}
                    onChange={(e) =>
                      handleFacilityChange(facility, e.target.value, "restaurant", true)
                    }
                  >
                  <option value="">Select</option>
                    {options.map((option, optIndex) => (
                      <option key={optIndex} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              )
            )}
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
