# import ale_py
# if using gymnasium
# import shimmy
import gymnasium as gym
import numpy as np

from gyn_fitness import GymFitnessFunction, SpaceInvadersFitness
from src.population import Population
from src.visualization import visualize_genome


def make_env():
    """ Makes a new gym 'CartPole-v1' environment. """
    return gym.make("CartPole-v1")


def make_env_lunar_lander():
    """ Makes a new gym LunarLander-v2 environment. """
    return gym.make("LunarLander-v2")


def make_car_env():
    """ Makes a new gym 'SpaceInvaders-ram-v0' environment. """
    return gym.make("SpaceInvaders-ram-v4", obs_type='ram')





if __name__ == '__main__':
    fitness_function = GymFitnessFunction(make_env=make_env_lunar_lander, default_num_episodes=5, default_max_steps=1000)
    population = Population(size=100, n_inputs=8, n_outputs=4,with_bias=True)
    
    population.evolve(
        generations=50,
        fitness_function=fitness_function
    )
    best_model = population.record_holder
    visualize_genome(best_model)

    env = gym.make("LunarLander-v2",render_mode='human')
    env.reset()
    action = env.action_space.sample()
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(action)
        h = best_model.apply(observation)
        action = (round(float(h[0])) if len(h) == 1
                                      else np.argmax(h))
        if terminated:
            break
        



    