import os
from pathlib import Path
import numpy as np
import random
import torch

def get_agent_id(agent_dir: str, env_name: str) -> str:
    """"""

    dir = Path(agent_dir)
    if not dir.exists():
        os.makedirs(dir)

    # try:
    #     agent_id = max([int(id) for id in os.listdir(dir)]) + 1
    # except ValueError:
    #     agent_id = 0

    ids = []
    for id in os.listdir(dir):
        try:
            ids.append(int(id))
        except:
            pass
        
    if len(ids) == 0:
        agent_id = 1
    else:
        agent_id = max(ids) + 1

    # stop()

    return str(agent_id)

def set_seed(
    env,
    seed
):
    """To ensure reproducible runs we fix the seed for different libraries"""
    random.seed(seed)
    np.random.seed(seed)

    env.seed(seed)
    env.action_space.seed(seed)

    torch.manual_seed(seed)

    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # env.seed(seed)
    # gym.spaces.prng.seed(seed)