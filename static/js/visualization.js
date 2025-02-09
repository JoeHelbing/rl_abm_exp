let socket = io();
let currentConfig = {};  // Will be populated when config loads
let gridCanvas = document.getElementById('gridCanvas');
let ctx = gridCanvas.getContext('2d');

// Separate data structures for plots
let happinessData = {
    episodes: [],
    type1: [],
    type2: []
};

let epsilonData = {
    episodes: [],
    type1: [],
    type2: []
};

// Load initial configuration before initializing plots
fetch('/config')
    .then(response => response.json())
    .then(config => {
        currentConfig = config;
        initializePlots();
    });

let happinessPlot = document.getElementById('happinessPlot');
let epsilonPlot = document.getElementById('epsilonPlot');

function initializePlots() {
    // Initialize happiness plot with threshold line
    Plotly.newPlot(happinessPlot, [
        {
            x: happinessData.episodes,
            y: happinessData.type1,
            type: 'scatter',
            name: 'Type 1 Agents'
        },
        {
            x: happinessData.episodes,
            y: happinessData.type2,
            type: 'scatter',
            name: 'Type 2 Agents'
        },
        {
            x: [0],
            y: [currentConfig.HAPPINESS_THRESHOLD],
            name: 'Happiness Threshold',
            type: 'scatter',
            mode: 'lines',
            line: {
                dash: 'dash',
                color: 'rgba(0,0,0,0.3)'
            }
        }
    ], {
        title: 'Current Happiness by Agent Type',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Happiness',
            range: [0, 1]
        }
    });

    // Initialize epsilon plot
    Plotly.newPlot(epsilonPlot, [
        {
            x: epsilonData.episodes,
            y: epsilonData.type1,
            type: 'scatter',
            name: 'Type 1 Agents'
        },
        {
            x: epsilonData.episodes,
            y: epsilonData.type2,
            type: 'scatter',
            name: 'Type 2 Agents'
        }
    ], {
        title: 'Exploration Rate Decay',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Epsilon',
            range: [0, 1]
        }
    });
}

function startSimulation() {
    // Clear previous data
    happinessData.episodes = [];
    happinessData.type1 = [];
    happinessData.type2 = [];
    epsilonData.episodes = [];
    epsilonData.type1 = [];
    epsilonData.type2 = [];
    
    // Get parameters from inputs
    const gridSize = parseInt(document.getElementById('gridSize').value);
    const numAgents = parseInt(document.getElementById('numAgents').value);
    const numEpisodes = parseInt(document.getElementById('numEpisodes').value);
    
    // Start simulation on server
    socket.emit('start_simulation', {
        grid_size: gridSize,
        num_agents: numAgents,
        num_episodes: numEpisodes
    });
}

function drawGrid(grid) {
    const canvas = gridCanvas;
    const width = canvas.width;
    const height = canvas.height;
    const cellSize = width / grid.length;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[i].length; j++) {
            if (grid[i][j] === 1) {
                ctx.fillStyle = '#2196F3';  // Blue for type 1
            } else if (grid[i][j] === 2) {
                ctx.fillStyle = '#F44336';  // Red for type 2
            } else {
                ctx.fillStyle = '#fff';     // White for empty
            }
            ctx.strokeStyle = '#ddd';
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
}

function updatePlots(episode, metrics) {
    // Create threshold line data
    const thresholdLine = {
        x: [episode],
        y: [currentConfig.HAPPINESS_THRESHOLD],
        name: 'Happiness Threshold',
        type: 'scatter',
        mode: 'lines',
        line: {
            dash: 'dash',
            color: 'rgba(0,0,0,0.3)'
        }
    };

    // Update happiness data
    happinessData.episodes.push(episode);
    happinessData.type1.push(metrics.average_happiness_type1);
    happinessData.type2.push(metrics.average_happiness_type2);
    
    // Update epsilon data
    epsilonData.episodes.push(episode);
    epsilonData.type1.push(metrics.average_epsilon_type1);
    epsilonData.type2.push(metrics.average_epsilon_type2);
    
    // Update both plots
    Plotly.update(happinessPlot, {
        x: [happinessData.episodes, happinessData.episodes, thresholdLine.x],
        y: [happinessData.type1, happinessData.type2, thresholdLine.y]
    });

    Plotly.update(epsilonPlot, {
        x: [epsilonData.episodes, epsilonData.episodes],
        y: [epsilonData.type1, epsilonData.type2]
    });
}

// Set up canvas size
function resizeCanvas() {
    const container = gridCanvas.parentElement;
    const size = Math.min(container.clientWidth, container.clientHeight);
    gridCanvas.width = size;
    gridCanvas.height = size;
}

// Handle window resize
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Socket event handlers
socket.on('state_update', function(data) {
    drawGrid(data.grid);
    updatePlots(data.episode, data.metrics);
});