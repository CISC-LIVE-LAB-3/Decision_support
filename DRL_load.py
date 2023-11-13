
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

standard_tank = StandardScaler()
standard_reactor = StandardScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScaleFactor:
    def __init__(self, lowVal, highVal):
        self.low_val = lowVal
        self.high_val = highVal

    def scale(self, value):        
        return (((value - 0) * (self.high_val - self.low_val)) / (1 - 0)) + self.low_val


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, n)
        self.l2 = nn.Linear(n, n)
        self.l3 = nn.Linear(n, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        af = self.l3(a)
        return torch.sigmoid(af)


class DRL_action:
    def __init__(self, state_dim=1, action_dim=1, n=64, load_path=None):
        self.actor = Actor(state_dim, action_dim, n).to(device)
        self.actor.load_state_dict(torch.load(load_path))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
   

standard_tank.fit_transform(sio.loadmat(r".\state_history_Tank.mat")["PSERB"][0][295:].reshape(-1, 1))
standard_reactor.fit_transform(sio.loadmat(r".\state_history_Reactor.mat")["TMAXREATORE"][0].reshape(-1, 1))

agent_DRL_S1_tank = DRL_action(load_path=r"./DRL_model_S1_tank_actor", n=64)
agent_DRL_S2_pump = DRL_action(load_path=r"./DRL_model_S2_pump_actor", n=256)
agent_DRL_S3_reactor = DRL_action(load_path=r"./DRL_model_S3_reactor_actor", n=64)

scaling_factor_tank = ScaleFactor(1.9, 5.4)
scaling_factor_pump = ScaleFactor(0.0, 0.7)
scaling_factor_reactor = ScaleFactor(100, 115)

'''usage'''
# from DRL_load import standard_tank, standard_reactor, agent_DRL_S1_tank, agent_DRL_S1_pump, agent_DRL_S1_reactor, scaling_factor_tank, scaling_factor_pump, scaling_factor_reactor

# action_tank = scaling_factor_tank.scale(agent_DRL_S1_tank.select_action(standard_tank.transform(np.array(["Pressure"]).reshape(-1,1))))
# action_pump = scaling_factor_pump.scale(agent_DRL_S2_pump.select_action((["Pressure"])))
# action_reactor = scaling_factor_reactor.scale(agent_DRL_S3_reactor.select_action(standard_reactor.transform(np.array(["MaxTemperature"]).reshape(-1,1))))


