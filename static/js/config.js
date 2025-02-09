let currentConfig = {};

async function loadCurrentConfig() {
    const response = await fetch('/config');
    currentConfig = await response.json();
    displayConfig();
}

function displayConfig() {
    const editor = document.getElementById('configEditor');
    editor.innerHTML = '';

    // Create sections for different parameter groups
    const sections = {
        'Agent Parameters': ['LEARNING_RATE', 'INITIAL_EPSILON', 'EPSILON_MIN', 'EPSILON_DECAY'],
        'Neural Network': ['NETWORK_LAYERS'],
        'RL Parameters': ['DISCOUNT_FACTOR', 'BATCH_SIZE', 'MEMORY_SIZE'],
        'Reward Parameters': ['REWARD_BONUS_THRESHOLD', 'REWARD_BONUS', 'REWARD_PENALTY_THRESHOLD', 'REWARD_PENALTY'],
        'Happiness Parameters': ['DIVERSITY_BONUS_THRESHOLD', 'DIVERSITY_BONUS']
    };

    for (const [section, params] of Object.entries(sections)) {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'config-section';
        sectionDiv.innerHTML = `<h3>${section}</h3>`;

        for (const param of params) {
            const value = currentConfig[param];
            const input = createConfigInput(param, value);
            sectionDiv.appendChild(input);
        }

        editor.appendChild(sectionDiv);
    }
}

function createConfigInput(param, value) {
    const container = document.createElement('div');
    container.className = 'config-input';

    const label = document.createElement('label');
    label.textContent = param + ': ';

    let input;
    if (Array.isArray(value)) {
        input = document.createElement('textarea');
        input.value = JSON.stringify(value, null, 2);
    } else if (typeof value === 'number') {
        input = document.createElement('input');
        input.type = 'number';
        input.step = value < 1 ? '0.001' : '1';
        input.value = value;
    } else {
        input = document.createElement('input');
        input.type = 'text';
        input.value = value;
    }

    input.onchange = () => updateConfigValue(param, input);
    
    container.appendChild(label);
    container.appendChild(input);
    return container;
}

function updateConfigValue(param, input) {
    let value = input.value;
    if (Array.isArray(currentConfig[param])) {
        try {
            value = JSON.parse(value);
        } catch (e) {
            alert('Invalid JSON format');
            input.value = JSON.stringify(currentConfig[param], null, 2);
            return;
        }
    } else if (typeof currentConfig[param] === 'number') {
        value = parseFloat(value);
    }
    
    currentConfig[param] = value;
    updateConfig();
}

async function updateConfig() {
    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentConfig)
        });
        const result = await response.json();
        if (result.status === 'error') {
            alert('Error updating config: ' + result.message);
        }
    } catch (e) {
        alert('Error updating config: ' + e.message);
    }
}

async function saveConfig() {
    const filename = document.getElementById('configFilename').value;
    try {
        const response = await fetch('/config/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename })
        });
        const result = await response.json();
        if (result.status === 'success') {
            alert('Configuration saved successfully');
        } else {
            alert('Error saving config: ' + result.message);
        }
    } catch (e) {
        alert('Error saving config: ' + e.message);
    }
}

async function loadConfig() {
    const filename = document.getElementById('configFilename').value;
    try {
        const response = await fetch('/config/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename })
        });
        const result = await response.json();
        if (result.status === 'success') {
            await loadCurrentConfig();
            alert('Configuration loaded successfully');
        } else {
            alert('Error loading config: ' + result.message);
        }
    } catch (e) {
        alert('Error loading config: ' + e.message);
    }
}

// Load configuration when page loads
document.addEventListener('DOMContentLoaded', loadCurrentConfig);