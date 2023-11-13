import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        af = self.l3(a)
        return torch.sigmoid(af)


class DRL_action:
    def __init__(self, state_dim=1, action_dim=1):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor.load_state_dict(torch.load(r"./DRL_model_S1_tank" + "_actor"))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


if __name__ == "__main__":
    '''Usage'''
    agent_DRL_S1_tank = DRL_action()
    agent_DRL_S1_tank.select_action([[["Pressure"]]])
