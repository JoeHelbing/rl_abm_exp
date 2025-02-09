let socket = io();
let currentConfig = {};  // Will be populated when config loads

// Separate data structures for each agent type
let type1Data = {};
let type2Data = {};

// Load initial configuration before initializing plots
fetch('/config')
    .then(response => response.json())
    .then(config => {
        currentConfig = config;
        initializePlots();  // Initialize plots after config is loaded
    });

// Initialize Plotly plots for each metric type
function initializePlots() {
    const thresholdLine = {
        x: [0],
        y: [currentConfig.HAPPINESS_THRESHOLD],
        name: 'Happiness Threshold',
        type: 'scatter',
        mode: 'lines',
        line: {
            dash: 'dash',
            color: 'rgba(0,0,0,0.3)'
        }
    };

    Plotly.newPlot('type1LossPlot', [], {
        title: 'Type 1 Agents Training Loss',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Loss', type: 'log' }
    });

    Plotly.newPlot('type2LossPlot', [], {
        title: 'Type 2 Agents Training Loss',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Loss', type: 'log' }
    });

    Plotly.newPlot('type1HappinessPlot', [thresholdLine], {
        title: 'Type 1 Agents Individual Happiness',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Happiness',
            range: [0, 1]
        }
    });

    Plotly.newPlot('type2HappinessPlot', [thresholdLine], {
        title: 'Type 2 Agents Individual Happiness',
        xaxis: { title: 'Episode' },
        yaxis: { 
            title: 'Happiness',
            range: [0, 1]
        }
    });
}

// Update plots with new data
function updatePlots(episode, agentMetrics) {
    // Sort metrics by agent type
    let type1Metrics = agentMetrics.filter(m => m.type === 1);
    let type2Metrics = agentMetrics.filter(m => m.type === 2);

    // Update type 1 agents data
    let type1Traces = type1Metrics.map(metric => {
        let agentId = metric.id;
        if (!type1Data[agentId]) {
            type1Data[agentId] = {
                episodes: [],
                loss: [],
                happiness: []
            };
        }
        type1Data[agentId].episodes.push(episode);
        type1Data[agentId].loss.push(metric.loss);
        type1Data[agentId].happiness.push(metric.happiness);
        
        return {
            x: type1Data[agentId].episodes,
            y: type1Data[agentId].loss,
            name: `Agent ${agentId}`,
            type: 'scatter',
            mode: 'lines'
        };
    });

    // Update type 2 agents data
    let type2Traces = type2Metrics.map(metric => {
        let agentId = metric.id;
        if (!type2Data[agentId]) {
            type2Data[agentId] = {
                episodes: [],
                loss: [],
                happiness: []
            };
        }
        type2Data[agentId].episodes.push(episode);
        type2Data[agentId].loss.push(metric.loss);
        type2Data[agentId].happiness.push(metric.happiness);
        
        return {
            x: type2Data[agentId].episodes,
            y: type2Data[agentId].loss,
            name: `Agent ${agentId}`,
            type: 'scatter',
            mode: 'lines'
        };
    });

    // Create threshold line for happiness plots
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

    // Update loss plots
    Plotly.react('type1LossPlot', type1Traces);
    Plotly.react('type2LossPlot', type2Traces);

    // Update happiness plots with threshold line
    Plotly.react('type1HappinessPlot', [
        ...type1Metrics.map(metric => ({
            x: type1Data[metric.id].episodes,
            y: type1Data[metric.id].happiness,
            name: `Agent ${metric.id}`,
            type: 'scatter',
            mode: 'lines'
        })),
        thresholdLine
    ]);

    Plotly.react('type2HappinessPlot', [
        ...type2Metrics.map(metric => ({
            x: type2Data[metric.id].episodes,
            y: type2Data[metric.id].happiness,
            name: `Agent ${metric.id}`,
            type: 'scatter',
            mode: 'lines'
        })),
        thresholdLine
    ]);
}

// Socket event handlers
socket.on('agent_metrics_update', function(data) {
    updatePlots(data.episode, data.agent_metrics);
});