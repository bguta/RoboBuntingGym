import numpy as np

import gymnasium as gym

import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module import RLModule

from robobunting_gym import RoboBuntingEnv


import os





def env_creator(config=None):
    env = RoboBuntingEnv(
    )
    return env


temp_env = RoboBuntingEnv()

obs_space = temp_env.observation_space["p1"]
act_space = temp_env.action_space["p1"]

ray.shutdown()
ray.init()

config = (
    PPOConfig()
    .environment(env=RoboBuntingEnv, 
                 disable_env_checking=True)
    .env_runners(num_env_runners=1) # used to collect samples 
    .framework("torch")
    .multi_agent(
        policies={
                "p1",
                "p2"
            },
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: agent_id),
        count_steps_by="env_steps",
    )
    .rl_module( # This is where we specify the RLModules to use for each policy.
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



# check that the checkpoint path exists and contains the expected files
assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"

# assert that the checkpoint path contains the expected files
expected_files = "learner_group/learner/rl_module"
assert os.path.exists(os.path.join(checkpoint_path, expected_files)), f"Checkpoint path {checkpoint_path} does not contain expected files: {expected_files}"

env = RoboBuntingEnv() # Use your custom class

modules = RLModule.from_checkpoint(
    checkpoint_path + "/learner_group" + "/learner" + "/rl_module"
    
)

obs, infos = env.reset()
done = False
max_steps_test = 2000
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

        # We have continuous actions, we sample the normal distribution. 
        # We could also use the mean (greedy action) by just taking action_dist_params[0:1] without sampling.
        # action_dist_params --> 0=mean, 1=log(stddev)
        sampled_action = np.clip(
            np.random.normal(action_dist_params[0:1], np.exp(action_dist_params[1:2])),
            a_min=env.action_space[agent_id].low[0],
            a_max=env.action_space[agent_id].high[0],
        )

        actions[agent_id] = sampled_action

    # Step the environment
    obs, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += rewards['p1']
    if (steps) % 10 == 0:
        env.render_screen()
    
    done = steps > max_steps_test

print(f"Total Reward: {total_reward/max_steps_test}")