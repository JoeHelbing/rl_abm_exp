from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from flask_socketio import SocketIO

from config import config
from simulation import Simulation

app = Flask(__name__)
app.secret_key = "schelling_rl_secret"
socketio = SocketIO(app)

# Global state
simulation = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration"""
    config_dict = {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and key.isupper()
    }
    return jsonify(config_dict)


@app.route("/config", methods=["POST"])
def update_config():
    """Update configuration"""
    try:
        new_config = request.json
        config._update_from_dict(new_config)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/config/save", methods=["POST"])
def save_config():
    """Save configuration to file"""
    try:
        filename = request.json.get("filename", "config.json")
        if not filename.endswith(".json"):
            filename += ".json"
        path = Path("configs") / filename
        path.parent.mkdir(exist_ok=True)
        config.save_to_file(str(path))
        return jsonify({"status": "success", "path": str(path)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/config/load", methods=["POST"])
def load_config():
    """Load configuration from file"""
    try:
        filename = request.json.get("filename")
        if not filename:
            return jsonify({"status": "error", "message": "No filename provided"}), 400
        if not filename.endswith(".json"):
            filename += ".json"
        path = Path("configs") / filename
        config.load_from_file(str(path))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def send_state_update(state):
    data = {
        "episode": state["episode"],
        "grid": state["grid"].tolist(),
        "metrics": state["metrics"],
    }
    socketio.emit("state_update", data)


@socketio.on("connect")
def handle_connect():
    if simulation and "is_running" in session and session.get("is_running", False):
        state = simulation.get_current_state()
        send_state_update(state)


@socketio.on("start_simulation")
def handle_start_simulation(data):
    global simulation

    session["is_running"] = True

    grid_size = data.get("grid_size", 10)
    num_agents_per_type = data.get("num_agents", 10)
    num_episodes = data.get("num_episodes", 100)

    simulation = Simulation(grid_size, num_agents_per_type)

    for _ in range(num_episodes):
        if not session.get("is_running", False):
            break

        metrics = simulation.run_episode()
        state = simulation.get_current_state()
        send_state_update(state)
        socketio.sleep(0.5)  # Small delay to control visualization speed

    session["is_running"] = False
    socketio.emit("simulation_stopped")


@socketio.on("stop_simulation")
def handle_stop_simulation():
    session["is_running"] = False


@socketio.on("disconnect")
def handle_disconnect():
    session["is_running"] = False


if __name__ == "__main__":
    socketio.run(app, debug=True)
