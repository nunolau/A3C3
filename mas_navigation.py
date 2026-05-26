#!/usr/bin/python3
"""
Multi-agent Navigation script.

Creates a GymNav environment and one thread per agent.  The main thread drives
the environment loop; each agent thread waits for its own observation, picks a
random action, and returns it to the main thread.  The loop runs until the
environment signals terminal=True.
"""

import sys
import queue
import random
import threading
import time

sys.path.insert(0, "simulator")
from GymNavigation import GymNav

# ── sentinel used to stop agent threads ──────────────────────────────────────
_STOP = object()


# ── Agent thread ──────────────────────────────────────────────────────────────
class AgentThread(threading.Thread):
    """One thread per agent.

    Communication with the main thread uses two queues:
      obs_queue  – main → agent  (observation ndarray or _STOP)
      act_queue  – agent → main  (integer action)
    """

    def __init__(self, agent_id: int, n_actions: int):
        super().__init__(name=f"agent-{agent_id}", daemon=True)
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.obs_queue: queue.Queue = queue.Queue()
        self.act_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Override this method to plug in a learned policy instead of random.
    # ------------------------------------------------------------------
    def select_action(self, observation) -> int:
        return random.randrange(self.n_actions)

    def run(self):
        print(f"[{self.name}] started")
        while True:
            obs = self.obs_queue.get()
            if obs is _STOP:
                print(f"[{self.name}] stopping")
                break
            action = self.select_action(obs)
            print(f"[{self.name}] obs={obs}  →  action={action}")
            self.act_queue.put(action)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(number_of_agents: int = 4, map_size: int = 15):
    env = GymNav(number_of_agents=number_of_agents, map_size=map_size)

    # Create one thread per agent
    agents = [
        AgentThread(agent_id=i, n_actions=env.agent_action_space)
        for i in range(number_of_agents)
    ]
    for agent in agents:
        agent.start()

    # ── reset ─────────────────────────────────────────────────────────
    observations, info = env.reset()
    print(f"[env] reset  observations={observations}  info={info}")

    # Send each agent its own initial observation
    for i, agent in enumerate(agents):
        agent.obs_queue.put(info["state_central"][i])

    step = 0
    while True:
        # Collect one action from every agent (blocks until each is ready)
        actions = [agent.act_queue.get() for agent in agents]

        env.render()

        observations, reward, terminal, info = env.step(actions)

        time.sleep(2)  # slow down for better visualization

        step += 1

        if terminal:
            print(f"\n[env] episode finished after {step} step(s).")
            break

        # Send next observation to each agent
        for i, agent in enumerate(agents):
            agent.obs_queue.put(info["state_central"][i])

    # ── shutdown agent threads ─────────────────────────────────────────
    for agent in agents:
        agent.obs_queue.put(_STOP)
    for agent in agents:
        agent.join()

    env.close()
    print("[env] done.")


if __name__ == "__main__":
    run(number_of_agents=2, map_size=4)
