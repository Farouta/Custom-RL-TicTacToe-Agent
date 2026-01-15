import paths

from src.environment import tictactoe
from src.agent import Agent
import random
import matplotlib.pyplot as plt

game1 = tictactoe()

agent1 = Agent() 
agent2 = Agent()
#if you have already trained agent2 against a bot with agent_bot_train.py you can use it as a teacher for the agent1
#agent2.load_model("./trained_agents/trained_agent2.pkl") 

num_games = 15000
batch_size = 1000

win_log, loss_log, draw_log = [], [], []
total_wins, total_losses, total_draws = 0, 0, 0


print("Starting Self-Play Training...")

for epoch in range(num_games):
    game1.reset_game()
    
    last_s1_canon, last_a1_canon = None, None
    last_s2_canon, last_a2_canon = None, None

    agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay_rate, agent1.epsilon_min)
    agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay_rate, agent2.epsilon_min)
    agent1.alpha = max(agent1.alpha * agent1.alpha_decay_rate, agent1.alpha_min)
    agent2.alpha = max(agent2.alpha * agent2.alpha_decay_rate, agent2.alpha_min)

    current_player = 1 if epoch % 2 == 0 else 2
    game1.current_player=current_player

    while not game1.gameover:
        
        # --- AGENT 1 TURN ---
        if current_player == 1:
            s1_canon, s1_transform = game1.get_canonical_key(game1.board)
            s1_canon_board = game1.key_state_to_board(s1_canon)
            a1_canon_possible = game1.get_possible_actions(s1_canon_board)
            
            a1_canon_chosen = agent1.choose_action(s1_canon, a1_canon_possible)
            
            if last_s1_canon is not None:
                agent1.update_q_table(last_s1_canon, last_a1_canon, 0.0, s1_canon, a1_canon_possible)

            a1_raw_chosen = game1.inverse_action(a1_canon_chosen, s1_transform)
            win_flag = game1.next_move(a1_raw_chosen[0], a1_raw_chosen[1])
            
            if game1.gameover:
                if win_flag == 1.0: # Agent 1 Wins
                    total_wins += 1  

                    agent1.update_q_table(s1_canon, a1_canon_chosen, 1.0, None, [])
                    if last_s2_canon is not None:
                        agent2.update_q_table(last_s2_canon, last_a2_canon, -1.0, None, [])
                else: # Draw
                    total_draws += 1
                    agent1.update_q_table(s1_canon, a1_canon_chosen, 0.0, None, [])
                    if last_s2_canon is not None:
                        agent2.update_q_table(last_s2_canon, last_a2_canon, 0.0, None, [])
                break
            
            last_s1_canon, last_a1_canon = s1_canon, a1_canon_chosen
            current_player = 2 

        # --- AGENT 2 TURN ---
        else: 
            s2_canon, s2_transform = game1.get_canonical_key(game1.board)
            s2_canon_board = game1.key_state_to_board(s2_canon)
            a2_canon_possible = game1.get_possible_actions(s2_canon_board)

            a2_canon_chosen = agent2.choose_action(s2_canon, a2_canon_possible)

            # Update Agent 2 for its PREVIOUS move
            if last_s2_canon is not None:
                agent2.update_q_table(last_s2_canon, last_a2_canon, 0.0, s2_canon, a2_canon_possible)
            
            a2_raw_chosen = game1.inverse_action(a2_canon_chosen, s2_transform)
            win_flag = game1.next_move(a2_raw_chosen[0], a2_raw_chosen[1])
            
            if game1.gameover:
                if win_flag == 2.0: # Agent 2 Wins
                    total_losses += 1 
                    agent2.update_q_table(s2_canon, a2_canon_chosen, 1.0, None, [])
                    if last_s1_canon is not None:
                        agent1.update_q_table(last_s1_canon, last_a1_canon, -1.0, None, [])
                else: # Draw
                    total_draws += 1
                    agent2.update_q_table(s2_canon, a2_canon_chosen, 0.0, None, [])
                    if last_s1_canon is not None:
                        agent1.update_q_table(last_s1_canon, last_a1_canon, 0.0, None, [])
                break

            last_s2_canon, last_a2_canon = s2_canon, a2_canon_chosen
            current_player = 1 
            
    if (epoch + 1) % batch_size == 0:
        win_rate = total_wins / batch_size
        loss_rate = total_losses / batch_size
        draw_rate = total_draws / batch_size
        
        win_log.append(win_rate)
        loss_log.append(loss_rate)
        draw_log.append(draw_rate)
        
        print(f"Epoch {epoch+1}/{num_games} | Win: {win_rate:.2f} | Loss: {loss_rate:.2f} | Draw: {draw_rate:.2f}")
        

        
        total_wins = 0
        total_losses = 0
        total_draws = 0

agent1.save_model("./trained_agents/trained_agent1.pkl")
agent2.save_model("./trained_agents/trained_agent2.pkl")
print("Models saved.")

print("Training complete!")

plt.figure(figsize=(12, 6))
plt.plot(win_log, label='AI Win Rate')
plt.plot(loss_log, label='AI Loss Rate')
plt.plot(draw_log, label='Draw Rate')
plt.xlabel(f'Batch of {batch_size} Games')
plt.ylabel('Rate')
plt.title('Phase 2: Optimization via Self-Play')
plt.legend()
plt.grid(True)
plt.show()