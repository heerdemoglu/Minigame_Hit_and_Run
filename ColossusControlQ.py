from pysc2.agents import base_agent
from pysc2.lib import actions, features
import math
import time
import numpy as np
import pandas as pd

# Action Definitions
_NO_OP = actions.FUNCTIONS.no_op.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

# Feature Definitions
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index

# Player Parameters
_PLAYER_SELF = 1

# Unit Indices
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

# Unit IDs
_ZERG_ZERGLING = 105
_PROTOSS_COLOSSUS = 4

_NOT_QUEUED = [0]
_QUEUED = [1]

# Unit Behavior Definitions
ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK = 'attack'
ACTION_MOVE = 'move'
ACTION_SELECT_ARMY = 'selectarmy'

action_set = [
        ACTION_DO_NOTHING,
        ACTION_ATTACK,
        ACTION_MOVE,
        ACTION_SELECT_ARMY,
]

# Reward Definitions -- !!IMPORTANT!!: MUST BE THE SAME AS IN MAP SCRIPT
KILL_UNIT_REWARD = 10
DEATH_REWARD = -50

# Implement Q-Learning
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, decay_rate=0.9, epsilon_greedy=0.9):
        # This function initializes the Q-Learning parameters
        
        self.action_list = actions
        self.lr = learning_rate
        self.y = decay_rate
        self.epsilon = epsilon_greedy
        self.table = pd.DataFrame(columns=self.action_list) # state / action pair list

    def choose_action(self, obs):
        # This function choses & returns an action to perform based on observation
        
        # Check if we're in a new state or in a previously visited one
        self.check_state(obs)

        # Coin Toss, decide to follow policy or to explore
        if np.random.uniform() < self.epsilon: # Folllow Current Policy
            # Choose the best action
            chosen_action = self.table.ix[obs, :]

            # Reindex actions with the same value
            chosen_action = chosen_action.reindex(np.random.permutation(chosen_action.index))
            
            # Choose the action with max value
            action = chosen_action.argmax()
        else: # Explore the environment
            # Choose a random action to perform
            action = np.random.choice(self.action_list)

        return action

    def learn(self, s_current, a, r, s_next):
        # This function implements the learning updates in terms of Q-Learning
        
        # Check if we ended up in a new state or in a previously visited one
        self.check_state(s_next)
        # Check if we ended up in a new state or in a previously visited one
        self.check_state(s_current)

        # Value function prediction for the current state / action pair
        prediction = self.table.ix[s_current, a]
        # Target calculation using alternate action under the same policy
        target = r + self.y * self.table.ix[s_next, :].max()

        # Q-Learning Update to the current state / action pair
        self.table.ix[s_current, a] += self.lr * (target - prediction)

    def check_state(self, state):
        # This function checks if the given state is a new one or a previously visited one
        # Keeps track of known states
        
        if state not in self.table.index:
            # Append new state to table
            self.table = self.table.append(pd.Series([0] * len(self.action_list), index=self.table.columns, name=state))


# Agent Code, subclasses BaseAgent, required to communicate with PYSC2
class ColossusControl(base_agent.BaseAgent):
    def __init__(self):
        # This function initializes all required parameters
        
        super(ColossusControl, self).__init__()
        
        self.Qlearn = QLearningTable(actions=list(range(len(action_set))))
        self.prev_unit_kill_score = 0
        self.prev_death_score = 0
        
        self.prev_action = None
        self.prev_state = None
        
    def step(self, obs):
        # This function performs a single step of the agent
        
        super(ColossusControl, self).step(obs)
        
        # Get the required observations from PYSC2
        unit_kill_score = obs.observation['score_cumulative'][5]
        death_score = obs.observation['score_cumulative'][3]
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        
        zergling_y, zergling_x = (player_relative == _ZERG_ZERGLING).nonzero()
        colossus_y, colossus_x = (player_relative == _PROTOSS_COLOSSUS).nonzero()

        # Define / update current state of the game using obversations
        current_state = [colossus_x,
                         colossus_y,
                         zergling_x,
                         zergling_y]
        
        # Choose an action from the Q-learning algorithm
        agent_action = self.Qlearn.choose_action(str(current_state))
        smart_action = action_set[agent_action]
        
        # Give immediate reward to the agent
        if self.prev_action is not None:
            reward = 0
            if unit_kill_score > self.prev_unit_kill_score:
                reward += KILL_UNIT_REWARD
            if death_score > self.prev_death_score:
                reward += DEATH_REWARD    
            # Give reward feedback to the agent
            self.Qlearn.learn(str(self.prev_state), self.prev_action, reward, str(current_state))
        
        # Time update to scores & states & actions
        self.prev_unit_kill_score = unit_kill_score
        self.prev_death_score = death_score
        self.prev_state = current_state
        self.prev_action = agent_action
        
        # Perform the Selected Action
        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP,[])
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])        
        elif smart_action == ACTION_MOVE:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                # Update position informations by performing another observation
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                colossus_y, colossus_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                zergling_y, zergling_x = (player_relative == _PLAYER_HOSTILE).nonzero()
                # Set default location if colossus not in screen & reduce colossus definition to a single point
                if math.isnan(colossus_x.mean()) or math.isnan(colossus_y.mean()):
                    player = [0, 0]
                else:
                    player = [int(colossus_x.mean()), int(colossus_y.mean())]
                colossus_x, colossus_y = player  
                closest, min_dist = None, None
                closest_x, closest_y = 0, 0
                # Get the closest zergling & its coords
                for p in zip(zergling_x, zergling_y):
                    dist = np.linalg.norm(np.array(player) - np.array(p))
                    if not min_dist or dist < min_dist:
                        closest, min_dist = p, dist
                        closest_x, closest_y = closest[0], closest[1]                               
                # Calculate point to move to run away from the closest zergling
                diff_x, diff_y = closest_x - colossus_x, closest_y - colossus_y
                step_size = 9
                if diff_x > 0:
                    target_x = closest_x - step_size
                else:
                    target_x = closest_x + step_size               
                if diff_y > 0:
                    target_y = closest_y - step_size
                else:
                    target_y = closest_y + step_size              
                # Check map boundaries
                if target_x > 83:
                    target_x = 83
                elif target_x < 0:
                    target_x = 0                    
                if target_y > 83:
                    target_y = 83
                elif target_y < 0:
                    target_y = 0                                    
                target = [target_x, target_y] 
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                # Update position informations by performing another observation
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                zergling_y, zergling_x = (player_relative == _PLAYER_HOSTILE).nonzero()
                # Check if any zergling exists
                if not zergling_y.any():
                    return actions.FunctionCall(_NO_OP, [])
                # Perform attack if any exists
                index = np.argmax(zergling_y)
                target = [zergling_x[index], zergling_y[index]]
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])

        return actions.FunctionCall(_NO_OP, [])

        time.sleep(0.05) # Limits Agent Action Capabilities
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])