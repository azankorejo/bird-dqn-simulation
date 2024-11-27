import random
import pybullet as p
import pybullet_data
import time
import math
import numpy as np

class Obstacle:
    def __init__(self, obstacle_type, position, size=None):
        self.obstacle_type = obstacle_type  # 'tree' or 'building'
        self.size = size if size else random.uniform(5, 15)  # Default size for tree or building
        self.position = [position[0], position[1], self.size / 2]  # Ensure obstacle is grounded

        # Create the obstacle based on its type
        if self.obstacle_type == 'tree':
            self.object_id = self.create_tree()
        elif self.obstacle_type == 'building':
            self.object_id = self.create_building()

    def create_tree(self):
        """Create a tree as a small cube."""
        tree_size = random.uniform(2, 4)  # Adjusted smaller size for trees
        tree_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tree_size, tree_size, tree_size])
        tree_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[tree_size, tree_size, tree_size],
                                             rgbaColor=[0.34, 0.24, 0.11, 1])  # Brown color
        return p.createMultiBody(baseMass=1, baseCollisionShapeIndex=tree_id, 
                                 baseVisualShapeIndex=tree_visual_id, basePosition=self.position)

    def create_building(self):
        """Create a building as a large cube."""
        building_size_x = random.uniform(10, 20)  # Larger width for buildings
        building_size_y = random.uniform(10, 20)
        building_size_z = random.uniform(20, 30)  # Varying height

        building_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[building_size_x / 2, building_size_y / 2, building_size_z / 2])
        building_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[building_size_x / 2, building_size_y / 2, building_size_z / 2],
                                                 rgbaColor=[0.7, 0.7, 0.7, 1])  # Light grey color

        # Position to be grounded properly
        position = [self.position[0], self.position[1], building_size_z / 2]
        return p.createMultiBody(baseMass=10, baseCollisionShapeIndex=building_id,
                                 baseVisualShapeIndex=building_visual_id, basePosition=position)

class ObstacleManager:
    def __init__(self, bird_position):
        self.obstacles = []
        self.bird_position = bird_position  # Bird's current position

    def generate_random_obstacles(self, num_obstacles=5, distance_range=(50, 100), area_size=(100, 100)):
        """Generate a list of random obstacles ahead of the bird, spaced out along the path."""
        for _ in range(num_obstacles):
            x = self.bird_position[0] + random.uniform(distance_range[0], distance_range[1])
            y = random.uniform(-area_size[0] / 2, area_size[0] / 2)
            height = random.uniform(5, 30)  # Varying height
            position = [x, y, height / 2]  # Ground position
            obstacle_type = random.choice(['tree', 'building'])
            obstacle = Obstacle(obstacle_type, position, size=height)
            self.obstacles.append(obstacle)

    def check_collision(self, bird_position):
        """Check if the bird is colliding with any obstacles."""
        for obstacle in self.obstacles:
            obs_pos = obstacle.position
            distance = math.sqrt((bird_position[0] - obs_pos[0])**2 + (bird_position[1] - obs_pos[1])**2)
            if distance < 5:  # Collision threshold
                return True
        return False