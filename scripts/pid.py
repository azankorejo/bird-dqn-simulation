import pybullet as p
import pybullet_data
import time
from pynput import keyboard
import math

# PID Class for controlling the bird's orientation
class PID:
    def __init__(self, Kp, Ki, Kd, max_output=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0
        self.max_output = max_output  # Max output to avoid excessive force

    def compute(self, setpoint, current_value):
        """
        Calculate the PID output for the given setpoint and current value.
        """
        error = setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        # Limit the output to prevent overcompensation
        return max(min(output, self.max_output), -self.max_output)