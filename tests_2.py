import math
import random

import imageio as imageio
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Function to simulate and visualize the SIR model
def graph_sir(S0, I0, R0, beta, gamma, days, S_sim, I_sim, R_sim, width, height, radius):
    # SIR model differential equations
    def sir_model(y, t, beta, gamma, radius, dimensions):
        S, I, R = y

        # probability any given person is within infection radius
        circle_probability = (math.pi * float(radius) * float(radius)) / float(dimensions[0] * dimensions[1])
        # recalculated beta based on the amount of infected people and probability
        accurate_beta = (1 - ((1 - beta) ** (circle_probability * I)))

        # expected change in populations in a timeframe (1 day)
        dS_dt = -accurate_beta * S
        dI_dt = accurate_beta * S - gamma * I
        dR_dt = gamma * I

        return [dS_dt, dI_dt, dR_dt]

    # Initial conditions: S0 (Susceptible), I0 (Infected), R0 (Recovered)
    initial_conditions = [S0, I0, R0]

    # Time points (days)
    t = np.linspace(0, days, days)

    # Solving the SIR model differential equations
    solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma, radius, (width, height)))
    S, I, R = solution.T

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible (Model)', color='blue', linestyle='--')
    plt.plot(t, I, label='Infected (Model)', color='red', linestyle='--')
    plt.plot(t, R, label='Recovered (Model)', color='green', linestyle='--')

    # Plotting the simulated data points with solid lines
    plt.plot(t, S_sim, label='Susceptible (Simulated)', color='blue', linestyle='-')
    plt.plot(t, I_sim, label='Infected (Simulated)', color='red', linestyle='-')
    plt.plot(t, R_sim, label='Recovered (Simulated)', color='green', linestyle='-')

    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIR Model vs Simulation')
    plt.legend()
    plt.grid(True)


def create_SIR_simulation(width=500, height=500, number_of_people=5000, num_infected=400, infect_rate=0.8,
                          recover_rate=0.15, infect_distance=10, max_move=5, simulation_name='simulation1'):
    people = []
    infected = []

    for _ in range(number_of_people):
        person = ["I" if random.random() < num_infected / number_of_people else "S",
                  random.randint(0, width), random.randint(0, height)]
        if person[0] == "I":
            infected.append(person)
        people.append(person)

    pygame.init()
    screen = pygame.display.set_mode((width, height))

    # Fill the screen with a background color (optional)
    screen.fill((0, 0, 0))  # Black background

    for p in people:
        # Draw the dot
        pygame.draw.circle(screen, (255, 0, 0) if p[0] == "I" else (0, 0, 255) if p[0] == 'S' else (0, 255, 0),
                           [p[1], p[2]], 1)

    # Update the display
    pygame.display.flip()

    def find_distance(p1, p2):
        x_diff = p1[1] - p2[1]
        y_diff = p1[2] - p2[2]
        return ((x_diff ** 2) + (y_diff ** 2)) ** 0.5

    def get_cell(x, y, cell_size):
        # Get the cell coordinates based on the person's position.
        return x // cell_size, y // cell_size

    def populate_grid(people, cell_size):
        # Organize people into a grid based on their positions.
        grid = {}
        for p in people:
            cell = get_cell(p[1], p[2], cell_size)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(p)
        return grid

    def step(people, infected, cell_size):
        new_infected = []
        recover = []
        grid = populate_grid(people, cell_size)

        for i in infected:
            cell = get_cell(i[1], i[2], cell_size)
            # Check the current cell and neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbor_cell in grid:
                        for p in grid[neighbor_cell]:
                            if p[0] == 'S' and find_distance(i, p) < infect_distance:
                                if random.random() < infect_rate:
                                    p[0] = 'I'
                                    new_infected.append(p)

            # Determine if the infected person recovers
            if random.random() < recover_rate:
                recover.append(i)
                i[0] = 'R'
        recovered_num = len(recover)
        # Remove recovered people from infected list
        for r in recover:
            infected.remove(r)

        return infected + new_infected, recovered_num

    def move_all(people):
        for p in people:
            p[1] += random.randint(max_move * -1, max_move)
            if p[1] < 1:
                p[1] = 1
            if p[1] > width - 1:
                p[1] = width - 1
            p[2] += random.randint(max_move * -1, max_move)
            if p[2] < 1:
                p[2] = 1
            if p[2] > height - 1:
                p[2] = height - 1

    frames = []
    step_count = 0
    simulated_S, simulated_I, simulated_R = [number_of_people - num_infected], [num_infected], [0]
    while len(infected) > 0:
        step_count += 1
        screen.fill((0, 0, 0))
        infected, recovered_amount = step(people, infected, infect_distance)
        move_all(people)
        for p in people:
            pygame.draw.circle(screen, (255, 0, 0) if p[0] == "I" else (0, 0, 255) if p[0] == 'S' else (0, 255, 0),
                               [p[1], p[2]], 1)

        pygame.display.flip()
        frame_array = pygame.surfarray.array3d(screen)  # Capture the frame
        frames.append(np.transpose(frame_array, (1, 0, 2)))  # Transpose for (height, width, color)

        pygame.time.delay(200)

        simulated_R.append(simulated_R[-1] + recovered_amount)
        simulated_I.append(len(infected))
        simulated_S.append(number_of_people - simulated_I[-1] - simulated_R[-1])
        pygame.image.save(screen, simulation_name + '.png')

    # Convert the frames to a video or GIF using imageio
    output_gif_path = "/tmp" + simulation_name + ".gif"
    imageio.mimsave(output_gif_path, frames, fps=10)
    print(f"Saved simulation as {output_gif_path}")

    # Simulate and visualize the SIR model
    graph_sir(number_of_people - num_infected, num_infected, 0, infect_rate, recover_rate, step_count + 1, simulated_S,
              simulated_I, simulated_R, width, height, infect_distance)

