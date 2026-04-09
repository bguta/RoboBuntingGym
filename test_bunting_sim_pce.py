import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
from pettingzoo import ParallelEnv

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import functools

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv


import glob
import os
import time

import pygame



WORLD_SIZE = 1
DIAMETER = 0.02
MAX_VELOCITY = 1
MAX_ACCELERATION = 1
MAX_STEPS = 20000
TIME_STEP = 0.001
SHADOW_DIST = 0.3

BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 1e-2
RENDER_INTERVAL = 20


class perceptualCrossingParallelEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "perceptual_crossing_v0", "render_fps": 30}

    def __init__(self, config=None,
                 world_size=WORLD_SIZE,
                 diameter=DIAMETER, 
                 max_steps=MAX_STEPS,
                 max_vel = MAX_VELOCITY,
                 max_acc = MAX_ACCELERATION,
                 time_step = TIME_STEP,
                 shadow_distance = SHADOW_DIST,
                 seed = None
                 ):
        super().__init__()

        self.render_mode = 'human'
        self.possible_agents = ["p1", "p2"]

        # configure environment
        self.world_size = world_size
        self.diameter = diameter
        self.max_steps = max_steps
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dt = time_step

        self.shadow_distance = shadow_distance
        self.hist_buffer = 20
        

        self.observation_space = {agent: gym.spaces.Box(low=-1, high=1, shape=(self.hist_buffer*4,), dtype=np.float32) for agent in self.possible_agents}
        self.action_space = {agent : gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) for agent in self.possible_agents}

        self.step_count = 0
    

      # rendering screen
        self.render_mode = 'human'
        self.screen = None
        self.clock = None
        self.canvas_size = 600
        self.ring_radius = 200
        self.agent_ui_size = 30 # Visual size in pixels

    def reset(self, *, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.pos = {agent: np.random.uniform(0, self.world_size) for agent in self.agents}
        self.vel = {agent: 0.0 for agent in self.agents}


        sensors = {a: 0.0 for a in self.agents}
        for i, a in enumerate(self.agents):
            other = self.agents[1 - i]

            targets = [
                self.pos[other],                               # the other 
                (self.pos[other] + self.shadow_distance) % self.world_size, # The shadow
            ]
            
            if any(self.__distance(self.pos[a], t) <= self.diameter for t in targets):
                sensors[a] = 1.0

        # memory buffer
        self.acc_hist       = {agent: [0.0]*self.hist_buffer for agent in self.agents}
        self.vel_hist       = {agent: [0.0]*self.hist_buffer for agent in self.agents}
        self.pos_hist       = {agent: [self.pos[agent]/self.world_size]*self.hist_buffer for agent in self.agents}
        self.contact_hist   = {agent: [sensors[agent]]*self.hist_buffer for agent in self.agents}

        # calculate reward
        self.internal_rewards = {agent : 0.0 for agent in self.agents}


        observations = self.__get_obs()
        infos = {agent: self.__get_info() for agent in self.agents}

        self.step_count = 0

        return observations, infos
    
    
    def step(self, actions):
        self.step_count += 1
        self.agents = self.possible_agents[:]

        # physics
        for agent in self.agents:
            acc = actions[agent][0] * self.max_acc
            self.vel[agent] = np.clip(self.vel[agent] + acc * self.dt, -self.max_vel, self.max_vel)
            self.pos[agent] = (self.pos[agent] + self.vel[agent] * self.dt) % self.world_size


        dist = self.__distance(self.pos[self.possible_agents[0]], self.pos[self.possible_agents[1]])
        overlap = 0
        if dist < self.diameter:
            overlap = 1 - dist/(self.world_size)

        sensors = {a: 0.0 for a in self.agents}
        for i, a in enumerate(self.agents):
            other = self.agents[1 - i]
            targets = [
                self.pos[other],                               # the other 
                (self.pos[other] + self.shadow_distance) % self.world_size, # The shadow
            ]
            
            if any(self.__distance(self.pos[a], t) <= self.diameter for t in targets):
                sensors[a] = 1.0

        # memory buffer
        for agent in self.agents:
            acc = actions[agent][0]
            self.acc_hist[agent]        = [acc]                             + self.acc_hist[agent][0:self.hist_buffer-1] # add new values to buffer

            self.vel_hist[agent]        = [self.vel[agent]/self.max_vel]    + self.vel_hist[agent][0:self.hist_buffer-1]
            self.pos_hist[agent]        = [self.pos[agent]/self.world_size] + self.pos_hist[agent][0:self.hist_buffer-1]
            self.contact_hist[agent]    = [sensors[agent]]                  + self.contact_hist[agent][0:self.hist_buffer-1]

        

        # observations
        observations = self.__get_obs()

        contact_shadow_t_ = {
            self.possible_agents[0] : self.__distance(self.pos[self.possible_agents[0]], (self.pos[self.possible_agents[1]] + self.shadow_distance) % self.world_size), # <= self.diameter,
            self.possible_agents[1] : self.__distance(self.pos[self.possible_agents[1]], (self.pos[self.possible_agents[0]] + self.shadow_distance) % self.world_size), #<= self.diameter
        }

        for agent in self.agents:

            contact_shadow_t = contact_shadow_t_[agent]
            contact_t =  dist#overlap
            energy_t = (self.vel[agent]*self.vel[agent])/(self.max_vel*self.max_vel)

            self.internal_rewards[agent] = overlap  - energy_t  #+ contact_t - contact_shadow_t - energy_t

        # reward contact
        rewards = self.internal_rewards

        # termination and info
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}

        terminations = {"__all__": self.step_count >= self.max_steps}
        infos = {agent: self.__get_info() for agent in self.agents}


        # if all(truncations.values()):
        #     self.agents = []

        return observations, rewards, terminations, truncations, infos
    
    def __distance(self, a, b):
        diff = abs(a - b)
        return min(diff, self.world_size - diff)
    

    def __get_obs(self):
        obs = {}
        self.agents = self.possible_agents[:]

        for agent in self.agents:
            obs[agent] = np.array(self.contact_hist[agent] + self.pos_hist[agent] + self.vel_hist[agent] + self.acc_hist[agent])

        return obs

    
    def __get_info(self):
        info = { # info for debugging
            'p1_pos': self.pos[self.possible_agents[0]],
            'p2_pos': self.pos[self.possible_agents[1]],
            'p1_v': self.vel[self.possible_agents[0]],
            'p2_v': self.vel[self.possible_agents[1]],
            'contact': self.__get_obs()[self.possible_agents[0]][0]
        }
        return info
    

    def render(self, bar_len=20):

        p1_s = (self.pos[self.possible_agents[0]] + self.shadow_distance) % self.world_size
        p2_s = (self.pos[self.possible_agents[1]] + self.shadow_distance) % self.world_size

        p1_idx = int((self.pos[self.possible_agents[0]]/ self.world_size) * (bar_len - 1))
        p2_idx = int((self.pos[self.possible_agents[1]] / self.world_size) * (bar_len - 1))

        p1_s_idx = int((p1_s/ self.world_size) * (bar_len - 1))
        p2_s_idx = int((p2_s/ self.world_size) * (bar_len - 1))

        line = ["."] * bar_len
        line[p1_idx] = "P1"
        line[p2_idx] = "P2"
        line[p1_s_idx] = "1s"
        line[p2_s_idx] = "2s"
        
        print("".join(line))

    def render_screen(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.canvas_size, self.canvas_size))
            pygame.display.set_caption("Perceptual Crossing Environment")
            self.clock = pygame.time.Clock()

        # 1. Reset Canvas
        canvas = pygame.Surface((self.canvas_size, self.canvas_size))
        canvas.fill((255, 255, 255))
        
        # 2. Draw Environment Ring
        pygame.draw.circle(canvas, (220, 220, 220), (self.canvas_size/2, self.canvas_size/2), self.ring_radius, width=2)

        # 3. Render Agents and Shadows
        colors = {"p1": (190, 18, 27), "p2": (25, 82, 184)}
        is_inside = {"p1": False, "p2": True}
        offset = {"p1": 0, "p2": 0}
        for agent_id in self.possible_agents:
            pos = self.pos[agent_id]
            color = colors[agent_id]
            
            # Agent Physics -> Circle Geometry
            angle = (pos / self.world_size) * 2 * np.pi
            coords = (self.ring_radius + offset[agent_id]) * np.array([np.cos(angle), np.sin(angle)])
            self.__draw_robot(canvas, coords, color, angle, inside=is_inside[agent_id])
            
            # Shadow Mapping
            s_pos = (pos + self.shadow_distance) % self.world_size
            s_angle = (s_pos / self.world_size) * 2 * np.pi
            s_coords = (self.ring_radius + offset[agent_id]) * np.array([np.cos(s_angle), np.sin(s_angle)])
            self.__draw_robot(canvas, s_coords, color, s_angle, is_shadow=True, inside=is_inside[agent_id])

        # 4. Update Display
        self.screen.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def __draw_robot(self, surface, center, color, angle, is_shadow=False, inside=False):
        """Internal helper to draw agents with sensors and rotation."""
        mid = np.array([self.agent_ui_size / 2, self.agent_ui_size / 2])
        agent_surf = pygame.Surface((self.agent_ui_size, self.agent_ui_size), pygame.SRCALPHA)
        
        # body & border
        alpha = 120 if is_shadow else 255
        pygame.draw.circle(agent_surf, (*color, alpha), mid, radius=self.agent_ui_size/2)
        pygame.draw.circle(agent_surf, (0, 0, 0), mid, radius=self.agent_ui_size/2, width=2)
        
        # sensor (yellow for real agent, Gray for shadow)
        sensor_color = (228, 213, 29) if not is_shadow else (150, 150, 150)
        pygame.draw.rect(agent_surf, sensor_color, [0, 0, self.agent_ui_size/3, self.agent_ui_size], border_radius=4)
        
        # rotate and blit to center
        ang_degree = np.degrees(-angle)
        if inside:
            ang_degree += 180
        
        rotated = pygame.transform.rotate(agent_surf, ang_degree)
        origin = np.array([self.canvas_size/2, self.canvas_size/2])
        blit_pos = (center + origin) - np.array(rotated.get_size()) / 2
        surface.blit(rotated, blit_pos)


    
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

def env_creator(config=None):
    env = perceptualCrossingParallelEnv(
    )
    return env


temp_env = perceptualCrossingParallelEnv()

obs_space = temp_env.observation_space["p1"]
act_space = temp_env.action_space["p1"]

ray.shutdown()
ray.init()

config = (
    PPOConfig()
    .environment(env=perceptualCrossingParallelEnv, 
                 disable_env_checking=True)
    .env_runners(num_env_runners=5) # used to collect samples 
    .framework("torch")
    .training(
        lr = LEARNING_RATE,
        train_batch_size_per_learner=BATCH_SIZE,
        num_epochs=10,
    )
    # .evaluation(
    #     evaluation_interval=2, # every n interval
    #     evaluation_num_env_runners=2,
    #     evaluation_duration_unit="episodes",
    #     evaluation_duration=100,
    # )
    .multi_agent(
        policies={
                "p1",
                "p2"
            },
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: agent_id),
        count_steps_by="env_steps",
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
            "p1" : RLModuleSpec(
                    model_config={"fcnet_hiddens": [128, 128]},
            ),
            "p2" : RLModuleSpec(
                 model_config={"fcnet_hiddens": [128, 128]},
            ) 
        }),
    )
)

algo = config.build()

checkpoint_path = os.getcwd() + "/raycheckpoint/"


import torch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.numpy import softmax



env = perceptualCrossingParallelEnv() # Use your custom class

modules = RLModule.from_checkpoint(
    checkpoint_path + "/learner_group" + "/learner" + "/rl_module"
    
)

obs, infos = env.reset()
done = False

max_steps_test = 10000
steps = 0
total_reward = 0
while not done:
    steps+=1
    actions = {}
    
    for agent_id, agent_obs in obs.items():
        # Identify which module to use
        module_id = "p1" if agent_id == "p1" else "p2"
        module = modules[module_id]
        
        # RLModules expect a batch. We wrap the single obs in a list/tensor.
        # Use 'forward_inference' for deterministic testing (no exploration)
        obs_batch = torch.tensor([agent_obs], dtype=torch.float32) 
        
        # Run the forward pass
        action_dist_params = module.forward_inference({"obs": obs_batch}, explortaion=False)['action_dist_inputs'].numpy()[0]

        #action = np.random.choice(env.action_space.n, p=softmax(action_dist_params[0]))
        # We have continuous actions, we take the mean (max likelihood).
        greedy_action = np.clip(
            action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space[agent_id].low[0],
            a_max=env.action_space[agent_id].high[0],
        )

        #action_dist = action_dist_class.from_logits(output['action_dist_inputs'])
        # act = action_dist.sample().numpy().item()
        actions[agent_id] = greedy_action

    # Step the environment
    obs, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += rewards['p1']
    if (steps) % 10 == 0:
        env.render_screen()
    
    done = steps > max_steps_test

print(f"Total Reward: {total_reward/max_steps_test}")
env.close()