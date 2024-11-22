const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 5000; // Define the port number

// Middleware
app.use(bodyParser.json()); // Parse JSON request bodies
app.use(cors()); // Enable CORS for all routes

// Example endpoint for receiving inputs
app.post('/process-inputs', (req, res) => {
    const userInputs = req.body; // Access the incoming data
    console.log('Received data from frontend:', userInputs);

    // Placeholder for ML processing or other logic
    const mockResults = {
        hotels: ['Hotel 1', 'Hotel 2', 'Hotel 3'],
        restaurants: ['Restaurant A', 'Restaurant B', 'Restaurant C'],
        attractions: ['Attraction X', 'Attraction Y', 'Attraction Z'],
    };

    // Send mock results back to the client
    res.json({ results: mockResults });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
