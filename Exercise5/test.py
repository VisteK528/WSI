import gymnasium
import random
import numpy as np
import matplotlib.pyplot as plt
import time

available_moves = [x for x in range(6)]
environment = gymnasium.make("Taxi-v3", render_mode="human")
environment.reset()

for i in range(100):
    environment.step(random.choice(available_moves))
    environment.render()
    time.sleep(0.1)