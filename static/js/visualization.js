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

let learningRateData = {
    episodes: [],
    values: []
};

// Data structures for loss facets
let type1LossData = {};
let type2LossData = {};

// Load initial configuration before initializing plots
fetch('/config')
    .then(response => response.json())
    .then(config => {
        currentConfig = config;
        initializePlots();
    });

let happinessPlot = document.getElementById('happinessPlot');
let epsilonPlot = document.getElementById('epsilonPlot');
let learningRatePlot = document.getElementById('learningRatePlot');
let type1LossFacets = document.getElementById('type1LossFacets');
let type2LossFacets = document.getElementById('type2LossFacets');

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

    // Initialize learning rate plot
    Plotly.newPlot(learningRatePlot, [
        {
            x: learningRateData.episodes,
            y: learningRateData.values,
            type: 'scatter',
            name: 'Learning Rate'
        }
    ], {
        title: 'Learning Rate Schedule',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Learning Rate',
            type: 'log'
        }
    });

    // Initialize loss facet plots with better handling of small values
    Plotly.newPlot(type1LossFacets, [], {
        title: 'Type 1 Agents Individual Loss',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Loss',
            type: 'log',
            exponentformat: 'e',
            range: [-7, 0]  // Show values from 10^-7 to 10^0
        },
        showlegend: true,
        height: 400
    });

    Plotly.newPlot(type2LossFacets, [], {
        title: 'Type 2 Agents Individual Loss',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Loss',
            type: 'log',
            exponentformat: 'e',
            range: [-7, 0]  // Show values from 10^-7 to 10^0
        },
        showlegend: true,
        height: 400
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
    learningRateData.episodes = [];
    learningRateData.values = [];
    
    // Clear loss facet data
    type1LossData = {};
    type2LossData = {};
    
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
    // Create threshold line data that spans the entire history
    const thresholdLine = {
        x: [Math.min(...happinessData.episodes, episode), episode],
        y: [currentConfig.HAPPINESS_THRESHOLD, currentConfig.HAPPINESS_THRESHOLD],
        name: 'Happiness Threshold',
        type: 'scatter',
        mode: 'lines',
        line: {
            dash: 'dash',
            color: 'rgba(0,0,0,0.3)'
        }
    };

    // Update data arrays
    happinessData.episodes.push(episode);
    happinessData.type1.push(metrics.average_happiness_type1);
    happinessData.type2.push(metrics.average_happiness_type2);
    epsilonData.episodes.push(episode);
    epsilonData.type1.push(metrics.average_epsilon_type1);
    epsilonData.type2.push(metrics.average_epsilon_type2);
    learningRateData.episodes.push(episode);
    learningRateData.values.push(metrics.average_lr);

    // Process metrics by agent type
    let type1Metrics = metrics.agent_metrics.filter(m => m.type === 1);
    let type2Metrics = metrics.agent_metrics.filter(m => m.type === 2);

    // Build type 1 agent traces with complete history
    const type1Traces = type1Metrics.map(metric => {
        let agentId = metric.id;
        if (!type1LossData[agentId]) {
            type1LossData[agentId] = { episodes: [], loss: [] };
        }
        type1LossData[agentId].episodes.push(episode);
        type1LossData[agentId].loss.push(metric.loss || 0);
        
        return {
            x: type1LossData[agentId].episodes,
            y: type1LossData[agentId].loss,
            name: `Agent ${agentId}`,
            type: 'scatter',
            mode: 'lines'
        };
    });

    // Build type 2 agent traces with complete history
    const type2Traces = type2Metrics.map(metric => {
        let agentId = metric.id;
        if (!type2LossData[agentId]) {
            type2LossData[agentId] = { episodes: [], loss: [] };
        }
        type2LossData[agentId].episodes.push(episode);
        type2LossData[agentId].loss.push(metric.loss || 0);
        
        return {
            x: type2LossData[agentId].episodes,
            y: type2LossData[agentId].loss,
            name: `Agent ${agentId}`,
            type: 'scatter',
            mode: 'lines'
        };
    });

    // Update all plots with complete histories
    Plotly.newPlot(happinessPlot, [
        {
            x: happinessData.episodes,
            y: happinessData.type1,
            name: 'Type 1 Agents',
            type: 'scatter',
            mode: 'lines'
        },
        {
            x: happinessData.episodes,
            y: happinessData.type2,
            name: 'Type 2 Agents',
            type: 'scatter',
            mode: 'lines'
        },
        thresholdLine
    ], {
        title: 'Current Happiness by Agent Type',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Happiness',
            range: [0, 1]
        }
    });

    Plotly.newPlot(epsilonPlot, [
        {
            x: epsilonData.episodes,
            y: epsilonData.type1,
            name: 'Type 1 Agents',
            type: 'scatter',
            mode: 'lines'
        },
        {
            x: epsilonData.episodes,
            y: epsilonData.type2,
            name: 'Type 2 Agents',
            type: 'scatter',
            mode: 'lines'
        }
    ], {
        title: 'Exploration Rate Decay',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Epsilon',
            range: [0, 1]
        }
    });

    Plotly.newPlot(learningRatePlot, [{
        x: learningRateData.episodes,
        y: learningRateData.values,
        name: 'Learning Rate',
        type: 'scatter',
        mode: 'lines'
    }], {
        title: 'Learning Rate Schedule',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Learning Rate',
            type: 'log'
        }
    });

    // Update loss facets with layout settings optimized for small values
    if (type1Traces.length > 0) {
        Plotly.newPlot(type1LossFacets, type1Traces, {
            title: 'Type 1 Agents Individual Loss',
            xaxis: { title: 'Episode' },
            yaxis: { 
                title: 'Loss',
                type: 'linear',
                exponentformat: 'e',
                showexponent: 'all',
                tickformat: '.1e'
            },
            showlegend: true,
            height: 400,
            margin: { l: 70, r: 40, t: 40, b: 50 }
        });
    }

    if (type2Traces.length > 0) {
        Plotly.newPlot(type2LossFacets, type2Traces, {
            title: 'Type 2 Agents Individual Loss',
            xaxis: { title: 'Episode' },
            yaxis: { 
                title: 'Loss',
                type: 'linear',
                exponentformat: 'e',
                showexponent: 'all',
                tickformat: '.1e'
            },
            showlegend: true,
            height: 400,
            margin: { l: 70, r: 40, t: 40, b: 50 }
        });
    }
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
    console.log('Received state update:', data); // Debug log
    console.log('Episode:', data.episode);
    if (data.metrics && data.metrics.agent_metrics) {
        console.log('Agent metrics received:', data.metrics.agent_metrics);
        const type1Count = data.metrics.agent_metrics.filter(m => m.type === 1).length;
        const type2Count = data.metrics.agent_metrics.filter(m => m.type === 2).length;
        console.log(`Found ${type1Count} type 1 agents and ${type2Count} type 2 agents`);
        
        // Log individual loss values
        data.metrics.agent_metrics.forEach(metric => {
            console.log(`Agent ${metric.id} (Type ${metric.type}): Loss = ${metric.loss}`);
        });
    } else {
        console.log('No metrics data in update');
    }
    drawGrid(data.grid);
    if (data.metrics) {
        updatePlots(data.episode, data.metrics);
    }
});