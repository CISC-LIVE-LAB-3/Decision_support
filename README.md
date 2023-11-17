# **Decision Support System Code**

This repository contains the code accompanying the journal paper **"Enhancing Control Room Operator Decision Making"** by J. Mietkiewicz et al. The code is designed to integrate with the Hugin API for its execution. It has been developed and tested using Hugin API version 9.2. If you are using a different version of the Hugin API, please update the import statement accordingly (e.g., `from pyhugin92 import *`).

## **Functionality**
The code primarily functions to identify faults within a system using the `model_Conflict.net` model. Following the detection of a fault, the Dynamic Influence Diagram `DID.oobn` model is employed to ascertain the most effective set of actions tailored to the current situation. The optimal recommendations are then outputted for the user.

## **Customization**
To adapt the model to specific scenarios or to modify its standard behavior, you can alter the functions `set_standard_evidence_conflict` and `set_standard_evidence_decision`. These functions are pivotal in setting the initial conditions and evidence for the model, thus influencing its decision-making process.

## **Requirements**
- **Hugin API (Version 9.2 and higher)**
- **Python environment**

Please ensure that your system meets these requirements for the successful execution of the code.
