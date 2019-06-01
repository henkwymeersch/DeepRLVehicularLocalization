# Deep RL for Vehicular Localization
Decentralized Scheduling for Cooperative Localization with Deep Reinforcement Learning

## Summary
This Python code learns and executes decentralized scheduling for cooperative localization of vehicles. Each link in the vehicular network is considered an agent and must decide whether or not to measure (using a radar-type measurement) based on the local network state. The goal is to localize all the vehicles as quickly as possible. Two methods are provided: a DQN implementation and a policy gradient (PG) implementation. We found that the PG version is more stable. 

## Usage
The code is used as follows (example for PG):
* Run train_PG to train the policy network. The file contains several parameters that can be modified by the user (lines 9-58).
* The trained model will be saved in the folder named "tf_models". Each model is saved in a separated folder. Name the folder "current" for the model you would like to test.
* Run test_PG to test the policy. The file also contains parameters that can be set to be different or the same as train_PG (i.e. "num_nds" and "num_lanes"). This allows centralized training on a small network and distributed testing on a larger network.
* Visualize the result with Environment.plot(). Use PEBs of all nodes as argument if you want to show PEBs in the figure.

## Authors
The code was authored by Dr. Bile Peng, while he was a Postdoctoral Researcher at Chalmers University of Technology. For questions, please contact <bile.peng@gmail.com>

## If you use this code
If you use or extend this code, please cite our work. 

B. Peng, G. Seco-Granados, E. Steinmetz, M. Fröhle and H. Wymeersch, "Decentralized Scheduling for Cooperative Localization With Deep Reinforcement Learning," in *IEEE Transactions on Vehicular Technology*, vol. 68, no. 5, pp. 4295-4305, May 2019.
doi: 10.1109/TVT.2019.2913695

```
@ARTICLE{PenSecSteFroWym:19,
  author={B. {Peng} and G. {Seco-Granados} and E. {Steinmetz} and M. {Frohle} and H. {Wymeersch}},
  journal={IEEE Transactions on Vehicular Technology},
  title={Decentralized Scheduling for Cooperative Localization with Deep Reinforcement Learning},
  year={2019},
  volume={},
  number={},
  pages={1-1},
  keywords={Reinforcement learning;Information exchange;Markov processes;Sensors;5G mobile communication;Wireless   communication;Machine-learning for vehicular localization;cooperative localization;deep reinforcement learning;deep   Qlearning;policy gradient},
  doi={10.1109/TVT.2019.2913695},
  ISSN={0018-9545},
  month={},}
```
