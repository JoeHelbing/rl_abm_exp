import logging
from asyncio import sleep
from json import dumps as json_dumps
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fasthtml.common import Div, P, Titled, fast_app, serve

from config import config
from simulation import Simulation

logging.basicConfig(level=logging.INFO)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create FastHTML app with websocket support
app, rt = fast_app(exts="ws")

# Global state
simulation = None


@rt("/")
def home():
    return Titled(
        "Schelling Model with RL Agents",
        Div(
            P("Welcome to the Schelling Model with RL Agents simulation."), id="content"
        ),
    )


@rt("/config", "GET")
def get_config():
    """Get current configuration"""
    config_dict = {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and key.isupper()
    }
    return json_dumps(config_dict)


@rt("/config", "POST")
def update_config(data):
    """Update configuration"""
    try:
        config._update_from_dict(data)
        return json_dumps({"status": "success"})
    except Exception as e:
        return json_dumps({"status": "error", "message": str(e)})


@rt("/config/save", "POST")
def save_config(data):
    """Save configuration to file"""
    try:
        filename = data.get("filename", "config.json")
        if not filename.endswith(".json"):
            return json_dumps({"status": "error", "message": "File must be .json"})

        path = Path("configs") / filename
        path.parent.mkdir(exist_ok=True)
        config.save_to_file(str(path))
        return json_dumps({"status": "success", "path": str(path)})
    except Exception as e:
        return json_dumps({"status": "error", "message": str(e)})


@rt("/config/load", "POST")
def load_config(data):
    """Load configuration from file"""
    try:
        filename = data.get("filename")
        if not filename:
            return json_dumps({"status": "error", "message": "No filename provided"})

        if not filename.endswith(".json"):
            return json_dumps({"status": "error", "message": "File must be .json"})

        path = Path("configs") / filename
        config.load_from_file(str(path))
        return json_dumps({"status": "success"})
    except Exception as e:
        return json_dumps({"status": "error", "message": str(e)})


async def send_state_update(state, send):
    data = {
        "episode": state["episode"],
        "grid": state["grid"].tolist(),
        "metrics": state["metrics"],
    }
    await send("state_update", json_dumps(data))


async def on_connect(send):
    logging.info("WebSocket connected")
    if simulation and app.session.get("is_running", False):
        state = simulation.get_current_state()
        await send_state_update(state, send)


async def on_disconnect(ws):
    logging.info("WebSocket disconnected")
    app.session["is_running"] = False


@app.ws("/ws", conn=on_connect, disconn=on_disconnect)
async def ws_handler(msg: dict, send):
    global simulation
    logging.info(f"Received message: {msg}")

    if msg.get("type") == "start_simulation":
        data = msg.get("data", {})
        app.session["is_running"] = True

        grid_size = data.get("grid_size", 10)
        num_agents_per_type = data.get("num_agents", 10)
        num_episodes = data.get("num_episodes", 100)

        simulation = Simulation(grid_size, num_agents_per_type)

        for episode in range(num_episodes):
            if not app.session.get("is_running", False):
                break

            state = simulation.step()
            state["episode"] = episode
            await send_state_update(state, send)
            await sleep(0.1)  # Small delay between steps

        app.session["is_running"] = False
        await send("simulation_stopped")

    elif msg.get("type") == "stop_simulation":
        app.session["is_running"] = False


if __name__ == "__main__":
    serve()
