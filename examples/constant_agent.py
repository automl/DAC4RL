"""
This is an example submission that can give participants a reference 
"""

# from dac4automlcomp.run_experiments import run_experiment

from dac4automlcomp.policy import DACPolicy
from rlenv.generators import DefaultRLGenerator, RLInstance

from carl.envs import *

import gym
import pdb

class ConstantPolicy(DACPolicy):
    '''
    A policy which checks the instance and applies the zoo parameters for PPO
    and applies them to the model. The parameters are based on the ones specified 
    in stable_baselines zoo 
    (https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)
    
    '''

    def __init__(self):
        """
        Initialize all the aspects needed for the policy. 
        """
        
        # Set the algorithm that will be used
        self.algorithm = 'PPO'
    

    def _get_zoo_params(self, env: str):
        """
        Return a set of hyperparameters for the given environment.

        Args:
            env: str
                The name of the environment.

        Returns:
            params: Dict
                The hyperparameters for the environment.
        """
        
        
        if env == 'CARLPendulumEnv':
            params = {
                "learning_rate" : 3e-4, 
                "gamma" : 0.99,
                "gae_lambda": 0.95,
                "vf_coef": 0.5,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            }
        elif env == 'CARLCartPoleEnv':
            params = {
                "learning_rate" : 0.001, 
                "gamma" : 0.98,
                "gae_lambda": 0.8,
                "vf_coef": 0.5,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            }
        elif env == 'CARLAcrobotEnv':
            params = {
                "learning_rate" : 0.0003, 
                "gamma" : 0.99,
                "gae_lambda": 0.94,
                "vf_coef": 0.5,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            }
        elif env == 'CARLMountainCarContinuousEnv':
            params = {
                "learning_rate" : 7.77e-05, 
                "gamma" : 0.9999,
                "gae_lambda": 0.9,
                "vf_coef": 0.19,
                "ent_coef": 0.00429,
                "clip_range": 0.1,
            }
        elif env == 'CARLLunarLanderEnv':
            params = {
                "learning_rate" : 0.0001, 
                "gamma" : 0.999,
                "gae_lambda": 0.98,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "clip_range": 0.2,
            }

        return params
    
    
    def act(self, state):
        """
        Generate an action in the form of the hyperparameters based on 
        the given instance 

        Args:
            state: Dict
                The state of the environment.

        Returns:
            action: Dict
        """
        # Get the environment from the state
        
        
        instance = state["instance"]

        # Check for compatibility
        assert isinstance(instance, RLInstance)

        # Extract information from the instance
        (env, context_features, context_std)= instance

        # Get the zoo parameters for hte environment
        zoo_params = self._get_zoo_params(env)

        # Create the action dictionary
        action = {"algorithm": "PPO"}
        action = {**action, **zoo_params}


        return action 

    def reset(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def seed(self):
        pass

if __name__ == "__main__":
    policy = ConstantPolicy()
    env = gym.make( "dac4carl-v0", 
                    total_timesteps=1e2, 
                    n_intervals=20
                )
    done = False
    state= None

    reward_history = []
    while not done:
        # get the default stat at reset
        state = env.reset()

        # generate an action
        action = policy.act(state)
        
        # Apply the action t get hte reward, next state and done
        state, reward, done, _ = env.step(action)
        
        #save the reward 
        reward_history.append(reward)
