import torch
import random
import numpy as np
import pickle

class Agent:
    def __init__(self):
        self.alpha=0.2
        self.alpha_decay_rate=0.9995
        self.alpha_min=0.0001
        self.gamma=0.95
        self.epsilon=0.3
        self.epsilon_decay_rate = 0.9995
        self.epsilon_min = 0.0
        self.q_table={} 
    def choose_action(self,state_key,possible_actions):
        explore=np.random.choice([1,0],p=[self.epsilon,1-self.epsilon])
        if explore :
            return random.choice(possible_actions)
        else :
            if state_key not in self.q_table:
                return random.choice(possible_actions)
            else:
                best_score=-999
                best_action=None
                action_scores=self.q_table[state_key]
                for action in possible_actions:
                    result=action_scores.get(action,0.0)
                    if best_score<result:
                        best_score=result
                        best_action=action
                return best_action    
    def update_q_table(self, state_key, action, reward, next_state_key, next_possible_actions):
        if state_key not in self.q_table:
            self.q_table[state_key]={}

        old_q_value=self.q_table[state_key].get(action, 0.0)
        best_future_score=0.0
        if next_possible_actions:

            if next_state_key in self.q_table:

                action_scores = self.q_table[next_state_key]
                scores = [action_scores.get(a, 0.0) for a in next_possible_actions]
                best_future_score = max(scores)


        error=reward+self.gamma*best_future_score -old_q_value
        new_q_value=old_q_value+self.alpha*error
        self.q_table[state_key][action]=new_q_value
    def save_model(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="q_table.pkl"):
        """Loads the Q-table from a file."""
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print("No saved model found! Starting with empty Q-table.")




