import pybullet as p
import pybullet_data
import time
from pynput import keyboard
import math
from pid import PID
import numpy as np
from obstacle import ObstacleManager

class HummingbirdSimulation:
    def __init__(self, model_path, texture_path, plane_texture_path):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load the plane and hummingbird models
        tile_size = 100  # Size of each tile
        num_tiles = 3   # Number of tiles in each direction

        # Load multiple planes and arrange them in a grid
        for x in range(-num_tiles, num_tiles):
            for y in range(-num_tiles, num_tiles):
                self.plane_id = p.loadURDF("plane.urdf", basePosition=[x * tile_size, y * tile_size, -1])        
                self.plane_texture_id = p.loadTexture(plane_texture_path)
                p.changeVisualShape(self.plane_id, -1, textureUniqueId=self.plane_texture_id)
        
        self.hummingbird_start_position = [0, 0, 2]  # Raise the initial position to ensure it's above the ground

        self.hummingbird_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.hummingbird_id = p.loadURDF(model_path, self.hummingbird_start_position, self.hummingbird_orientation)
        self.texture_id = p.loadTexture(texture_path)

        # Apply the texture to the model (assuming the model has the right mapping)
        p.changeVisualShape(self.hummingbird_id, -1, textureUniqueId=self.texture_id)
        p.resetBasePositionAndOrientation(self.hummingbird_id, [0, 0, 1], p.getQuaternionFromEuler([3.14159, 0, 0]))  # 180-degree rotation
        p.changeDynamics(self.hummingbird_id, -1, angularDamping=0.05)  # Tune as needed

        # Movement forces (scaling can be adjusted)
        self.directional_forces = {
            "forward": [0, -50, 0],
            "backward": [0, 50, 0],
            "right": [-100, 0, 0],
            "left": [100, 0, 0],
            "up": [0, 0, 50],
            "down": [0, 0, -50]
        }

        self.tilt_forces = {
            "left": [0, 3, 0],   # Apply torque for left horizontal tilt (around the y-axis)
            "right": [0, -3, 0], # Apply torque for right horizontal tilt (around the y-axis)
        }
        self.smooth_turn_forces = {
            "left": [5, -5, 0],    # Apply force to move left smoothly (in a curved path)
            "right": [5, 5, 0],    # Apply force to move right smoothly (in a curved path)
        }

        self.current_direction = None
        self.active_keys = set()  # Set to store active keys


        self.pitch_pid = PID(Kp=10.0, Ki=0.1, Kd=0.5, max_output=10.0)
        self.roll_pid = PID(Kp=10.0, Ki=0.1, Kd=0.5, max_output=10.0)
        self.yaw_pid = PID(Kp=10.0, Ki=0.1, Kd=0.5, max_output=10.0)

        self.camera_distance = 10  # Distance of camera behind the bird (can be changed)
        self.camera_height = 2.0  # Height of camera from bird (adjustable)

        self.obstacle_manager = ObstacleManager(self.get_state()[:3])
        self.obstacle_manager.generate_random_obstacles()

    def get_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.hummingbird_id)
        velocity, _ = p.getBaseVelocity(self.hummingbird_id)
        return np.array([position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]])


    def take_action(self, action):
        """
        Perform the action (move the bird based on the action taken).
        Actions are mapped to [forward, backward, left, right, up, down].
        """
        position, _ = p.getBasePositionAndOrientation(self.hummingbird_id)
        
        if action == 0:  # forward
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["forward"], position, p.WORLD_FRAME)
        elif action == 1:  # backward
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["backward"], position, p.WORLD_FRAME)
        elif action == 2:  # left
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["left"], position, p.WORLD_FRAME)
        elif action == 3:  # right
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["right"], position, p.WORLD_FRAME)
        elif action == 4:  # up
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["up"], position, p.WORLD_FRAME)
        elif action == 5:  # down
            p.applyExternalForce(self.hummingbird_id, -1, self.directional_forces["down"], position, p.WORLD_FRAME)

    def step(self, action):
        self.take_action(action)
        self.apply_forces()
        self.update_camera_position()  # Update camera to follow the bird
        p.stepSimulation()
        state = self.get_state()

        # Reward System
        reward = 0.0
        done = False

        if state[2] < 1.0:  # Ground penalty
            reward -= 2
            done = True
        elif abs(state[0]) > 100 or abs(state[1]) > 100:  # Out-of-bounds penalty
            reward -= 20
            done = True
        elif self.obstacle_manager.check_collision(state[:3]):  # Obstacle collision penalty
            reward -= 10
            done = True
        else:
            reward += 2  # Survival reward
            min_distance = self.get_min_distance_to_obstacle(state[:3])
            if min_distance < 2.0:  # Proximity penalty
                reward -= 20 / min_distance
            reward -= 0.1 * abs(state[5])  # Smooth flight penalty
            reward += 1e-3  # Time increment reward

        return state, reward, done

    def get_min_distance_to_obstacle(self, bird_position):
        """Calculate minimum distance from the bird to any obstacle."""
        min_distance = float('inf')
        for obstacle in self.obstacle_manager.obstacles:
            obs_pos = np.array(obstacle.position)
            distance = np.linalg.norm(np.array(bird_position) - obs_pos)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def update_camera_position(self):
            # Get current position and orientation of the bird
            position, orientation = p.getBasePositionAndOrientation(self.hummingbird_id)
            
            # Convert the orientation to Euler angles to calculate the direction
            pitch, roll, yaw = self.get_euler_from_quaternion(orientation)
            
            # Update camera to follow the bird
            p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=yaw, cameraPitch=-160, cameraTargetPosition=[position[0], position[1], position[2]])


    def apply_initial_force(self):
        # Apply an initial upward force to keep the bird afloat
        initial_force = [0, 0, 10]  # Adjust upward force to ensure it doesn't fall immediately
        p.applyExternalForce(self.hummingbird_id, -1, initial_force, [0, 0, 0], p.WORLD_FRAME)

    def update_position(self):
        # Get current position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.hummingbird_id)
        return position, orientation

    def apply_forces(self):
        # Get current position and orientation of the bird
        position, orientation = p.getBasePositionAndOrientation(self.hummingbird_id)
        current_pitch, current_roll, current_yaw = self.get_euler_from_quaternion(orientation)

        # Stabilize orientation using PID controllers
        pitch_output = self.pitch_pid.compute(0, current_pitch)  # Target pitch = 0 (level)
        roll_output = self.roll_pid.compute(0, current_roll)  # Target roll = 0 (level)
        yaw_output = self.yaw_pid.compute(0, current_yaw)  # Target yaw = 0 (level)

        # Apply corrective forces for stabilization
        correction_force = [pitch_output, roll_output, yaw_output]
        p.applyExternalForce(self.hummingbird_id, -1, correction_force, position, p.WORLD_FRAME)

        # Apply directional forces based on all active keys
        for key in self.active_keys:
            if key in self.directional_forces:
                movement_force = self.directional_forces[key]
                p.applyExternalForce(self.hummingbird_id, -1, movement_force, position, p.WORLD_FRAME)
                if key in self.tilt_forces:
                    p.applyExternalTorque(self.hummingbird_id, -1, self.tilt_forces[key], p.WORLD_FRAME)

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.active_keys.add("forward")
            elif key == keyboard.Key.down:
                self.active_keys.add("backward")
            elif key == keyboard.Key.left:
                self.active_keys.add("left")
            elif key == keyboard.Key.right:
                self.active_keys.add("right")
            elif key == keyboard.Key.home:
                self.active_keys.add("down")
            elif key == keyboard.Key.insert:
                self.active_keys.add("up")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up:
                self.active_keys.discard("forward")
            elif key == keyboard.Key.down:
                self.active_keys.discard("backward")
            elif key == keyboard.Key.left:
                self.active_keys.discard("left")
            elif key == keyboard.Key.right:
                self.active_keys.discard("right")
            elif key == keyboard.Key.home:
                self.active_keys.discard("down")
            elif key == keyboard.Key.insert:
                self.active_keys.discard("up")
        except AttributeError:
            pass
        
    def get_euler_from_quaternion(self, quat):
        """
        Converts quaternion (w, x, y, z) to Euler angles (pitch, roll, yaw).
        """
        w, x, y, z = quat
        pitch = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        roll = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        return pitch, roll, yaw
