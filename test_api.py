from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from tests_2 import create_SIR_simulation
import os
import time
import random
import string

app = FastAPI()

# Paths to the generated files
GIF_PATH = "simulation.gif"
FRAME_PATH = "current_frame.png"

simulations = []
simulations_lock = False


# Function to delete files after 5 minutes
def delete_files_after_delay(simulation_id: str):
    """Delete files"""
    time.sleep(30)
    gif_path = f"{simulation_id}.gif"
    frame_path = f"{simulation_id}.png"

    # Delete GIF and frame if they exist
    if os.path.exists(gif_path):
        os.remove(gif_path)
    if os.path.exists(frame_path):
        os.remove(frame_path)

    # Remove from simulation start times
    simulations.remove(simulation_id)
    print(f"Files for {simulation_id} have been deleted.")


def generate_unique_simulation_id():
    """Generate a unique simulation ID."""
    global simulations_lock
    while simulations_lock:
        time.sleep(1)
    simulations_lock = True
    while True:
        # Generate a random 8-character string
        simulation_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        if simulation_id not in simulations:
            simulations_lock = False
            return simulation_id


@app.post("/api/simulation/start")
async def start_simulation(background_tasks: BackgroundTasks):
    """
    Endpoint to start the simulation and generate the GIF.
    """
    simulation_id = generate_unique_simulation_id()
    # reserve the simulation id
    simulations.append(simulation_id)

    # Remove old files if they exist
    if os.path.exists(GIF_PATH):
        os.remove(GIF_PATH)
    # Start the background task to create the simulation
    create_SIR_simulation(simulation_name=simulation_id)

    sim_gif_path = "/tmp" + simulation_id + '.gif'
    # Start a background task to delete files after some time
    background_tasks.add_task(delete_files_after_delay, simulation_id)
    return FileResponse(sim_gif_path, media_type='image/gif')

