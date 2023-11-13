import sys
from pyhugin92 import *
from DRL_load import standard_tank, standard_reactor, agent_DRL_S1_tank, agent_DRL_S2_pump, agent_DRL_S3_reactor, scaling_factor_tank, scaling_factor_pump, scaling_factor_reactor
import numpy as np
#
# Function Definitions
#



# Main Execution

Pressure=101000

RL_value=scaling_factor_pump.scale(agent_DRL_S2_pump.select_action(([Pressure/101350])))

print(RL_value)

RL_value=scaling_factor_tank.scale(agent_DRL_S1_tank.select_action(standard_tank.transform(np.array([Pressure/101350]).reshape(-1,1))))
print(RL_value)
agent_DRL = agent_DRL_S2_pump
RL_value = scaling_factor_pump.scale(agent_DRL_S2_pump.select_action(([Pressure/ 101350])))

print(RL_value)