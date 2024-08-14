from src.fun_agent import FuN
from src.fun_agent import fixedSizeList

from src.dqn import DQN

import torch

from torchinfo import summary

import numpy as np

from torchviz.dot import make_dot


show_attributes = False

input = torch.rand(2).to(torch.float32)

state_list = fixedSizeList(11)

#while not state_list.is_at_max_capacity():
#    state_list.push(torch.rand(2))

model = FuN(d=128,
            n_actions=4,
            k=8,
            c=1,
            r=1,
            manager_goal_eps=0.2,
            run_on_gridworld=True,
            is_ram=False,
            device='cpu').to('cpu')

dqn = DQN(4).to('cpu')

#percept = model.percept
#worker = model.worker
#manager = model.manager 

summary(model, input_size=input.shape)

summary(dqn, input_size=input.shape)

# Percept section
#with torch.no_grad():
#    percept_output = percept(input)

#with torch.no_grad():
#make_dot(manager(percept_output, state_list, 0.9), show_attrs=show_attributes, params=dict(manager.named_parameters())).render('manager_dot', format='png')

#with torch.no_grad():
#make_dot(worker(percept_output, torch.tensor(np.random.rand(10, 256)).to(torch.float32)), show_attrs=show_attributes, params=dict(worker.named_parameters())).render('worker_dot', format='png')

