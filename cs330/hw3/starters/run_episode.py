"""Run the Q-network on the environment for fixed steps.

Complete the code marked TODO."""
import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import


def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state, goal_state)
        episodic_return (float): reward collected during the episode
        succeeded (bool): DQN succeeded to reach the goal state or not
    """

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False
    episodic_return = 0.0

    # reset the environment to get the initial state
    state, goal_state = env.reset() # pylint: disable=unused-variable

    for _ in range(steps_per_episode):

        # ======================== TODO modify code ========================
        pass

        # append goal state to input, and prepare for feeding to the q-network
        state_goal=torch.tensor(np.append(state,goal_state),dtype=torch.float32)
        # forward pass to find action
        actions=q_net(state_goal)
        action=actions.argmax()
        
        # take action, use env.step
            #         state (ndarray): new state_vector of size (num_bits,)
            # reward (float): 0 if state != goal and 1 if state == goal
            # done (bool): value indicating if the goal has been reached
            # info (dict): dictionary with the goal state and success boolean
            
        nextstate,reward,done,info=env.step(action)
        # add transition to episode_experience as a tuple of
        # (state, action, reward, next_state, goal)
        episode_experience.append((state,action,reward,nextstate,goal_state))
        # update episodic return
        episodic_return+=reward
        # update state
        state=nextstate.copy()
        # update succeeded bool from the info returned by env.step
        if info["successful_this_state"]:
            succeeded=True
        if done:break
        # break the episode if done=True

        # ========================      END TODO       ========================

    return episode_experience, episodic_return, succeeded
