#!/usr/bin/python3
"""
Multi-agent Navigation script with inter-agent communication.

Like mas_navigation.py, but:
  - Each agent only receives its own local observation (not state_central).
  - Agents broadcast a fixed-size message vector every step; every other
    agent receives those messages at the *next* step, matching the convention
    used in Navigation/MA3CSlave.py.

Communication model (mirrors Navigation/MA3CSlave.py):
  - At step t each agent outputs (action, message[comm_size]).
  - At step t+1 each agent receives a concatenation of the messages sent
    by every *other* agent in index order, giving a received vector of
    shape [(n_agents - 1) * comm_size].
  - Optional delivery failure replaces a dropped message with zeros.
"""

import sys
import queue
import random
import threading
import time
from typing import List

import numpy as np

sys.path.insert(0, "simulator")
from GymNavigation import GymNav

# ── sentinel used to stop agent threads ──────────────────────────────────────
_STOP = object()


# ── Agent thread ──────────────────────────────────────────────────────────────
class AgentThread(threading.Thread):
    """One thread per agent.

    Communication with the main thread uses two queues:
      obs_queue  – main → agent  (tuple (observation, received_messages) or _STOP)
      act_queue  – agent → main  (tuple (int action, np.ndarray message))

    received_messages is a 1-D ndarray of shape
        [(n_agents - 1) * comm_size]
    concatenating messages from every other agent in ascending index order
    (the agent's own index is skipped), matching the MA3C convention.

    The sent message is a 1-D ndarray of shape [comm_size].
    """

    def __init__(self, agent_id: int, n_agents: int, n_actions: int, comm_size: int):
        super().__init__(name=f"agent-{agent_id}", daemon=True)
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.comm_size = comm_size
        self.obs_queue: queue.Queue = queue.Queue()
        self.act_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Override these two methods to plug in a learned policy.
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray, received_messages: np.ndarray) -> int:
        """Return an integer action given local obs and received messages."""
        return random.randrange(self.n_actions)

    def make_message(self, observation: np.ndarray, received_messages: np.ndarray) -> np.ndarray:
        """Return a comm_size float vector to broadcast to other agents."""
        return np.zeros(self.comm_size, dtype=np.float32)

    def run(self):
        print(f"[{self.name}] started")
        while True:
            item = self.obs_queue.get()
            if item is _STOP:
                print(f"[{self.name}] stopping")
                break
            observation, received_messages = item
            action = self.select_action(observation, received_messages)
            message = self.make_message(observation, received_messages)
            print(
                f"[{self.name}] obs={observation}  recv={received_messages}"
                f"  →  action={action}  msg={message}"
            )
            self.act_queue.put((action, message))


# ── Communication helpers ─────────────────────────────────────────────────────
def build_received_messages(
    sent: List[np.ndarray],
    agent_id: int,
    comm_delivery_failure_chance: float = 0.0,
) -> np.ndarray:
    """Build the received-message vector for *agent_id*.

    Concatenates messages from all agents except agent_id.  A message from
    agent j is replaced with zeros if delivery fails (sampled per message).
    Returns a 1-D array of shape [(n_agents - 1) * comm_size].
    """
    parts = []
    for j, msg in enumerate(sent):
        if j == agent_id:
            continue
        if comm_delivery_failure_chance > 0.0 and random.random() < comm_delivery_failure_chance:
            parts.append(np.zeros_like(msg))
        else:
            parts.append(msg.copy())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(
    number_of_agents: int = 4,
    map_size: int = 15,
    comm_size: int = 2,
    comm_delivery_failure_chance: float = 0.0,
):
    env = GymNav(number_of_agents=number_of_agents, map_size=map_size)

    # Create one thread per agent
    agents = [
        AgentThread(
            agent_id=i,
            n_agents=number_of_agents,
            n_actions=env.agent_action_space,
            comm_size=comm_size,
        )
        for i in range(number_of_agents)
    ]
    for agent in agents:
        agent.start()

    # Initialise zero messages for the very first step (no prior communication)
    zero_msgs = [np.zeros(comm_size, dtype=np.float32) for _ in range(number_of_agents)]

    # ── reset ─────────────────────────────────────────────────────────
    observations, info = env.reset()
    print(f"[env] reset  observations={observations}  info={info}")

    # Send each agent its own local observation + zero messages
    for i, agent in enumerate(agents):
        recv = build_received_messages(zero_msgs, i, comm_delivery_failure_chance)
        agent.obs_queue.put((observations[i], recv))

    step = 0
    while True:
        # Collect one (action, message) from every agent (blocks until each is ready)
        results = [agent.act_queue.get() for agent in agents]
        actions = [r[0] for r in results]
        sent_messages = [r[1] for r in results]

        env.render()

        print(f"\n[env] step {step}  actions={actions}")
        observations, reward, terminal, info = env.step(actions)
        print(f"[env] reward={reward}  terminal={terminal}")

        time.sleep(2)  # slow down for better visualisation

        step += 1

        if terminal:
            print(f"\n[env] episode finished after {step} step(s).")
            break

        # Distribute sent messages and send next local observation to each agent
        for i, agent in enumerate(agents):
            recv = build_received_messages(sent_messages, i, comm_delivery_failure_chance)
            agent.obs_queue.put((observations[i], recv))

    # ── shutdown agent threads ─────────────────────────────────────────
    for agent in agents:
        agent.obs_queue.put(_STOP)
    for agent in agents:
        agent.join()

    env.close()
    print("[env] done.")


if __name__ == "__main__":
    run(number_of_agents=2, map_size=4, comm_size=2)
