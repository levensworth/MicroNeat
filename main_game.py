import gymnasium as gym
import numpy as np

from gyn_fitness import GymFitnessFunction
from src.population import Population
from src.visualization import visualize_genome





def make_env():
    """ Makes a new gym 'CartPole-v1' environment. """
    return gym.make("CartPole-v1")




if __name__ == '__main__':
    fitness_function = GymFitnessFunction(make_env=make_env, default_num_episodes=5, default_max_steps=500)
    population = Population(size=100, n_inputs=4, n_outputs=2, with_bias=True)
    
    population.evolve(
        generations=5,
        fitness_function=fitness_function
    )
    best_model = population.fittest()
    # visualize_genome(best_model)

    env = gym.make("CartPole-v1",render_mode='human')
    env.reset()
    action = action = env.action_space.sample()
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(action)
        h = best_model.apply(observation)
        action = (round(float(h[0])) if len(h) == 1
                                      else np.argmax(h))
        if terminated:
            break
        



    