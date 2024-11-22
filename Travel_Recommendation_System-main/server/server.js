
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios');  // Import axios to make HTTP requests

const app = express();
const PORT = 5000; // Define the port number

// Middleware
app.use(bodyParser.json()); // Parse JSON request bodies
app.use(cors()); // Enable CORS for all routes

// Example endpoint for receiving inputs
app.post('/process-inputs', async (req, res) => {
    const userInputs = req.body; // Access the incoming data
    console.log('Received data from frontend:', userInputs);
    
    // Prepare the data to send to the Flask server (recommendation.py)
    const { city, state, hotelFacilities, restaurantFacilities } = userInputs;
    // console.log('Restaurant Fac:', restaurantFacilities);
    
    try {
        // Make a request to the Flask API for hotel recommendations
        const response = await axios.post('http://localhost:5001/recommendations', {
            city,
            state,
            hotelFacilities,
            restaurantFacilities
        });

        // Get the recommendations from Flask API response
        const recommendations = response.data;

        // Send the recommendations back to the client
        res.json({ results: recommendations });
    } catch (error) {
        console.error('Error while calling Flask API:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
