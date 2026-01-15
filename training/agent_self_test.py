from src.environment import tictactoe
from src.agent import Agent
import random
import matplotlib.pyplot as plt

game1 = tictactoe()

# Agent 1: The "Learner" 
agent1 = Agent() 
agent1.load_model("trained_agent2.pkl") 

agent2 = Agent()
agent2.load_model("trained_agent2.pkl") 

num_games = 50000
batch_size = 1000

win_log, loss_log, draw_log = [], [], []
total_wins, total_losses, total_draws = 0, 0, 0

# Settings
agent1.epsilon = 0.0  
agent2.epsilon = 0.0

print("Starting Self-Play Training...")

for epoch in range(num_games):
    game1.reset_game()
    

    current_player = 1 if epoch % 2 == 0 else 2
    game1.current_player=current_player

    while not game1.gameover:
        
        # --- AGENT 1 TURN ---
        if current_player == 1:
            s1_canon, s1_transform = game1.get_canonical_key(game1.board)
            s1_canon_board = game1.key_state_to_board(s1_canon)
            a1_canon_possible = game1.get_possible_actions(s1_canon_board)
            
            a1_canon_chosen = agent1.choose_action(s1_canon, a1_canon_possible)
            

            a1_raw_chosen = game1.inverse_action(a1_canon_chosen, s1_transform)
            win_flag = game1.next_move(a1_raw_chosen[0], a1_raw_chosen[1])
            
            if game1.gameover:
                if win_flag == 1.0: # Agent 1 Wins
                    total_wins += 1  

                else: # Draw
                    total_draws += 1
                break
            
            last_s1_canon, last_a1_canon = s1_canon, a1_canon_chosen
            current_player = 2 

        # --- AGENT 2 TURN ---
        else: 
            s2_canon, s2_transform = game1.get_canonical_key(game1.board)
            s2_canon_board = game1.key_state_to_board(s2_canon)
            a2_canon_possible = game1.get_possible_actions(s2_canon_board)

            a2_canon_chosen = agent2.choose_action(s2_canon, a2_canon_possible)

            
            a2_raw_chosen = game1.inverse_action(a2_canon_chosen, s2_transform)
            win_flag = game1.next_move(a2_raw_chosen[0], a2_raw_chosen[1])
            
            if game1.gameover:
                if win_flag == 2.0: # Agent 2 Wins
                    total_losses += 1 
                else: # Draw
                    total_draws += 1
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


print("Training complete!")

plt.figure(figsize=(12, 6))
plt.plot(win_log, label='AI Win Rate')
plt.plot(loss_log, label='AI Loss Rate')
plt.plot(draw_log, label='Draw Rate')
plt.xlabel(f'Batch of {batch_size} Games')
plt.ylabel('Rate')
plt.title('Validation vs. Self (100% Draw)')
plt.legend()
plt.grid(True)
plt.show()