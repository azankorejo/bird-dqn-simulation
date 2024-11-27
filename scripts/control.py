import pybullet as p
import time
from pynput import keyboard
from obstacle import ObstacleManager


class HummingbirdControl:
    def __init__(self, simulation):
        self.simulation = simulation
        obstacle_manager = ObstacleManager(bird_position=[0, 0, 10])
        obstacle_manager.generate_random_obstacles(num_obstacles=50, area_size=(200, 200))


    def run(self):
        # Keyboard listener for controlling movement
        listener = keyboard.Listener(on_press=self.simulation.on_press, on_release=self.simulation.on_release)
        listener.start()

        # Main simulation loop
        try:
            while True:
                # Apply forces for stabilization and movement
                self.simulation.apply_forces()
                self.simulation.update_camera_position()  # Update camera to follow the bird
                # Step the simulation
                p.stepSimulation()
                time.sleep(1 / 240)  # Maintain a steady simulation speed

        except KeyboardInterrupt:
            print("Simulation ended by user.")
        finally:
            p.disconnect()
            listener.stop()