"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent


import matplotlib.pyplot as plt
import imageio
from typing import Union


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        
        total_reward += r
        s = next_s
        
        if done:
            break
        # END SOLUTION

    return total_reward

def play_and_train_sarsa(env: gym.Env, agent: SarsaAgent, t_max=int(1e4)) -> float:
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    a = agent.get_action(s)

    for _ in range(t_max):
        next_s, r, done, _, _ = env.step(a)
        
        if done:
            agent.update(s, a, r, next_s, None)
            total_reward += r
            break
            
        next_a = agent.get_action(next_s)
        agent.update(s, a, r, next_s, next_a)
        
        total_reward += r
        s = next_s
        a = next_a

    return total_reward


#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################
# Test Q-Learning
q_learning_rewards = []
q_agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

for i in range(1000):
    reward = play_and_train(env, q_agent)
    q_learning_rewards.append(reward)
    if i % 100 == 0:
        print(f"Q-Learning Episode {i}, Mean Reward: {np.mean(q_learning_rewards[-100:])}")


# Test Q-Learning with Epsilon Scheduling
q_eps_rewards = []
q_eps_agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=1.0, gamma=0.99, legal_actions=list(range(n_actions)),
    epsilon_end=0.05, epsilon_decay_steps=500
)

for i in range(1000):
    reward = play_and_train(env, q_eps_agent)
    q_eps_rewards.append(reward)
    if i % 100 == 0:
        print(f"Q-Learning (ε-scheduling) Episode {i}, Mean Reward: {np.mean(q_eps_rewards[-100:])}")



####################
# 3. Play with SARSA
####################

# Test SARSA
sarsa_rewards = []
sarsa_agent = SarsaAgent(
    learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))
)

for i in range(1000):
    reward = play_and_train_sarsa(env, sarsa_agent)
    sarsa_rewards.append(reward)
    if i % 100 == 0:
        print(f"SARSA Episode {i}, Mean Reward: {np.mean(sarsa_rewards[-100:])}")

""" def display_agent_behavior(agent: Union[QLearningAgent, SarsaAgent], episodes: int = 3):
    display_env = gym.make("Taxi-v3", render_mode="human")
    for episode in range(episodes):
        state, _ = display_env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.get_action(state)
            state, reward, done, truncated, _ = display_env.step(action)
            total_reward += reward
            
        print(f"Episode {episode + 1} terminé avec une récompense totale de {total_reward}")
    
    display_env.close()

print("\nDémonstration de l'agent Q-Learning :")
display_agent_behavior(q_agent)

print("\nDémonstration de l'agent Q-Learning avec Epsilon Scheduling :")
display_agent_behavior(q_eps_agent)

print("\nDémonstration de l'agent SARSA :")
display_agent_behavior(sarsa_agent)
 """
 
def create_video(env: gym.Env, agent: Union[QLearningAgent, SarsaAgent], filename: str):
    # Create a temporary environment with rgb_array render mode
    tmp_env = gym.make("Taxi-v3", render_mode="rgb_array")
    
    # Wrap the environment with RecordVideo
    video_env = gym.wrappers.RecordVideo(
        env=tmp_env, 
        video_folder="./videos",
        name_prefix=filename,
        episode_trigger=lambda _: True  # Record every episode
    )
    
    s, _ = video_env.reset()
    done = False
    
    while not done:
        a = agent.get_action(s)
        s, _, done, _, _ = video_env.step(a)
    
    video_env.close()
    print(f"Video saved as {filename} in ./videos directory")

# Usage in your main code:
# Make sure to create a 'videos' directory first
import os
os.makedirs("./videos", exist_ok=True)

# Then use the function for each agent
create_video(env, q_agent, "q_learning")
create_video(env, q_eps_agent, "q_learning_eps")
create_video(env, sarsa_agent, "sarsa")

plt.figure(figsize=(10, 6))
plt.plot(np.convolve(q_learning_rewards, np.ones(100)/100, mode='valid'), label='Q-Learning')
plt.plot(np.convolve(q_eps_rewards, np.ones(100)/100, mode='valid'), label='Q-Learning (ε-scheduling)')
plt.plot(np.convolve(sarsa_rewards, np.ones(100)/100, mode='valid'), label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Average Reward (100 episodes)')
plt.title('Comparison of RL Algorithms on Taxi-v3')
plt.legend()
plt.savefig('comparison.png')
plt.close()

""" 
rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
# TODO: créer des vidéos de l'agent en action




agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action




agent = SARSAAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
 """