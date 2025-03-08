const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    console.log('Health check requested');
    res.json({
        status: 'running',
        message: 'Backend is working'
    });
});

// Prediction endpoint
app.post('/api/predict', async (req, res) => {
    console.log('\n=== New Prediction Request ===');
    console.log('Request body:', JSON.stringify(req.body, null, 2));
    
    try {
        console.log('Spawning Python process...');
        console.log('Current directory:', __dirname);
        console.log('Python script path:', path.join(__dirname, 'predict.py'));

        const pythonCommand = process.env.NODE_ENV === 'production' ? 'python3' : 'python';
        console.log('Python command:', {pythonCommand});
        
        const pythonProcess = spawn(pythonCommand, [
            'predict.py',
            JSON.stringify(req.body)
        ], {
            cwd: __dirname
        });

        let result = '';
        let error = '';



        pythonProcess.stdout.on('data', (data) => {
            console.log('Python stdout:', data.toString());
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error('Python stderr:', data.toString());
            error += data.toString();
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            
            if (code !== 0) {
                console.error('Python process error details:', error);
                return res.status(500).json({
                    status: 'error',
                    error: 'Failed to process prediction',
                    details: error
                });
            }

            try {
                console.log('Raw Python output:', result);
                const prediction = JSON.parse(result);
                console.log('Parsed prediction:', prediction);
                res.json(prediction);
            } catch (e) {
                console.error('Error parsing prediction result:', e);
                console.error('Failed to parse:', result);
                res.status(500).json({
                    status: 'error',
                    error: 'Failed to parse prediction result',
                    details: e.message
                });
            }
        });

        pythonProcess.on('error', (err) => {
            console.error('Failed to start Python process:', err);
            res.status(500).json({
                status: 'error',
                error: 'Failed to start prediction process',
                details: err.message
            });
        });

    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({
            status: 'error',
            error: error.message,
            stack: error.stack
        });
    }
});

// Start server
app.listen(port, () => {
    console.log(`\n=== Server Started ===`);
    console.log(`Time: ${new Date().toISOString()}`);
    console.log(`Port: ${port}`);
}); 