import sys
from pyhugin92 import *
from DRL_load import standard_tank, standard_reactor, agent_DRL_S1_tank, agent_DRL_S2_pump, agent_DRL_S3_reactor, scaling_factor_tank, scaling_factor_pump, scaling_factor_reactor
import numpy as np
#
# Function Definitions
#
#sdshsfthsgfhstgjbalkjfbgælnhgl
def parse_listener(line, description):
    """A listener for parsing errors."""
    print(f"Parse error line {line}: {description}")

def load_model(cc_name, class_name, steps):
    """Load a Bayesian network model."""
    cc = ClassCollection()
    try:
        cc.parse_classes(cc_name, parse_listener)
        return Domain(cc.get_class_by_name(class_name), steps)
    except HuginException:
        print("A Hugin Exception was raised!")
        raise

def set_evidence(node, value):
    """Set evidence for a node."""

    state_index = node.get_state_index_from_label(value) if isinstance(value, str) else node.get_state_index_from_value(value)
    node.select_state(state_index)

def find_least_conflicting_state(domain, fault_node):
    """Determine the least conflicting state for a fault node."""
    min_conflict = float('inf')
    least_conflicting_state = []

    for state in range(fault_node.get_number_of_states()):
        set_standard_evidence_conflict(domain)
        fault_node.select_state(state)
        domain.propagate()
        current_conflict = domain.get_conflict()
        domain.initialize()

        if current_conflict < min_conflict:
            min_conflict = current_conflict
            least_conflicting_state = [fault_node.get_name(), fault_node.get_state_label(state)]

    return least_conflicting_state

def set_standard_evidence_conflict(domain):
    """Set standard evidence for the domain."""
    evidence_values = {
        "PressureT1": 101000,
        "Nitrogen_flow": 0,
        # If auto is 1 it will detect first a control fault. If it 0 will detect valve fault
        "Auto":1,
        "Systeme": 1
    }
    for node_name, value in evidence_values.items():
        set_evidence(domain.get_node_by_name(node_name), value)


def set_standard_evidence_decision(domain):
    """Set standard evidence for the domain."""
    evidence_values = {
        "T0.PressureT1": 101000,
        "T0.Level": 6.4999,
        "T0.Fault": "none",
    }
    for node_name, value in evidence_values.items():
        set_evidence(domain.get_node_by_name(node_name), value)


def find_best_decision(domain, decision_nodes):
    """Find the best decision for each node."""
    L_best_decision = []
    nitrogen_flow=domain.get_node_by_name("T1.Nitrogen_flow")

    for decision_node in decision_nodes:
        utility_max = -float('inf')
        for state in range(decision_node.get_number_of_states()):

            if decision_node.get_name() == "Systeme" and state == 1:
                value = 0
                state = domain.get_node_by_name("set_point_Nitrogen").get_state_index_from_value(value)
                nitrogen_flow.select_state(state)

            utility = decision_node.get_expected_utility(state)
            if utility >= utility_max:
                utility_max = utility
                right_state = state
            nitrogen_flow.retract_findings()
        decision_node.select_state(right_state)
        domain.propagate()
        L_best_decision.append([decision_node.get_name(),decision_node.get_state_value(right_state)])

    return(L_best_decision)


# Main Execution - Conflict Detection
#

conflict_model_name = "model_Conflict"
conflict_cc_name = f"C:\\Users\\jomie\\Documents\\GitHub\\Code_Journal_processes\\{conflict_model_name}.net"

conflict_dom = Domain.parse_domain(conflict_cc_name )

conflict_dom.compile()

fault_node = conflict_dom.get_node_by_name("Fault")
least_conflicting_state = find_least_conflicting_state(conflict_dom, fault_node)
#print(least_conflicting_state)

conflict_dom.delete()

#
# Main Execution - Decision Making


#

decision_model_name = "DID"
decision_cc_name = f"C:\\Users\\jomie\\Documents\\GitHub\\Code_Journal_processes\\{decision_model_name}.oobn"

decision_dom = load_model(decision_cc_name, decision_model_name, 10)
decision_dom.compile()
decision_dom.compress()

decision_node_names = ["T1.Auto_Nitrogen", "T1.Set_point_Nitrogen", "T1.Systeme", "T1.Auto_Pump", "T1.Set_point_pump"]
decision_nodes = [decision_dom.get_node_by_name(name) for name in decision_node_names]

set_evidence(decision_dom.get_node_by_name(f"T1.{least_conflicting_state[0]}"), least_conflicting_state[1])


set_standard_evidence_decision(decision_dom)

decision_dom.propagate()
best_decisions = np.array(find_best_decision(decision_dom, decision_nodes))




Pressure=101000


def adjust_decision_for_fault_control(pressure, scaling_factor, agent, transformer, decision, node):
    """
    Adjust the decision based on the output of the DRL agent for 'Fault control'.
    """
    normalized_pressure = transformer.transform(np.array([pressure / 101350]).reshape(-1, 1))
    rl_value = scaling_factor.scale(agent.select_action(normalized_pressure))[0]

    value = float(decision[1])
    index = node.get_state_index_from_value(value)
    uper_interval = node.get_state_value(index + 1)

    if rl_value < uper_interval and value < rl_value:
        decision[1] = rl_value

    return decision

def adjust_decision_for_fault_valve(pressure, scaling_factor, agent, decision, node):
    """
    Adjust the decision based on the output of the DRL agent for 'Fault valve'.
    """
    normalized_pressure = pressure / 101350
    rl_value = np.round(scaling_factor.scale(agent.select_action([normalized_pressure]))[0],2)
    value=float(decision[1])
    index=node.get_state_index_from_value(value)

    uper_interval=node.get_state_value(index+1)

    if rl_value < uper_interval and value<rl_value:
        decision[1] = rl_value

    return decision

# Main Execution
Pressure = 101000

if least_conflicting_state[1] == "Fault control":
    node=decision_dom.get_node_by_name(best_decisions[1][0])
    best_decisions[1] = adjust_decision_for_fault_control(
        Pressure, scaling_factor_tank, agent_DRL_S1_tank,
        standard_tank, best_decisions[1], node
    )

elif least_conflicting_state[1] == "Fault valve":
    node=decision_dom.get_node_by_name(best_decisions[4][0])
    best_decisions[4] = adjust_decision_for_fault_valve(
        Pressure, scaling_factor_pump, agent_DRL_S2_pump,
        best_decisions[4], node
    )


#print(best_decisions)

decision_dom.delete()



data = best_decisions
# Iterate over the list and process the conditions
for i in range(len(data)):
    if data[i][0] == 'T1.Auto_Nitrogen' and data[i][1] == '0.0':

        set_point_nitrogen = data[i + 1][1] if i + 1 < len(data) else "Unknown"
        print(f"Set the nitrogen flow to manual mode with a set point of {set_point_nitrogen}")

    if data[i][0] == 'T1.Systeme' and data[i][1] == '2.0':
        print("Switch to backup system")

    if data[i][0] == 'T1.Auto_Pump' and data[i][1] == '0.0':

        set_point_pump = data[i + 1][1] if i + 1 < len(data) else "Unknown"
        print(f"Set the pump power to auto with a set the point of {set_point_pump}")























