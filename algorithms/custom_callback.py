from stable_baselines3.common.callbacks import BaseCallback
from env.custom_hopper import *

class myCallback(BaseCallback):
    def __init__(self, train_env, width: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.width = width
        self.env = train_env
        self.episode_starts = True

    def _on_step(self) -> bool:
        if self.locals['dones']:
            self.env.set_random_parameters(self.width)
        return True
