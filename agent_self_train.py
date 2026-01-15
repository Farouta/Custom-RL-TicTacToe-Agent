from src.environment import tictactoe
from src.agent import Agent
import random
import matplotlib.pyplot as plt

game1=tictactoe()
agent2=Agent()
agent1=Agent()
agent2.load_model("trained_agent2.pkl")




num_games = 10000
win_log = []
loss_log = []
draw_log = []
total_wins = 0
total_losses = 0
total_draws = 0
sub_count = 1000
agent2.epsilon = 0.0
for epoch in range(num_games + 1):
    game1.reset_game()
    last_s1_canon, last_a1_canon = None, None
    last_s2_canon, last_a2_canon = None, None

    agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay_rate, agent1.epsilon_min)
    agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay_rate, agent2.epsilon_min)

    current_player = 1

    while not game1.gameover:
        if current_player == 1:
            s1_canon, s1_transform = game1.get_canonical_key(game1.board)
            s1_canon_board = game1.key_state_to_board(s1_canon)
            a1_canon_possible = game1.get_possible_actions(s1_canon_board)
            
            a1_canon_chosen = agent1.choose_action(s1_canon, a1_canon_possible)
            
            if last_s1_canon is not None:
                agent1.update_q_table(last_s1_canon, last_a1_canon, 0.0, s1_canon, a1_canon_possible)

            a1_raw_chosen = game1.inverse_action(a1_canon_chosen, s1_transform)

            r1 = game1.next_move(a1_raw_chosen[0], a1_raw_chosen[1])
            
            s1_next_canon, _ = game1.get_canonical_key(game1.board)

            if game1.gameover:
                if r1 == 1.0: 
                    total_wins +=1  
                    agent1.update_q_table(s1_canon, a1_canon_chosen, 1.0, None, [])
                    #if last_s2_canon is not None:
                        #agent2.update_q_table(last_s2_canon, last_a2_canon, -1.0, s1_next_canon, [])
                else:  
                    total_draws +=1
                    agent1.update_q_table(s1_canon, a1_canon_chosen, -0.2,None, [])
                    #if last_s2_canon is not None:
                        #agent2.update_q_table(last_s2_canon, last_a2_canon, 0.0, s1_next_canon, [])
                break
            
            last_s1_canon, last_a1_canon = s1_canon, a1_canon_chosen
            
            current_player = 2 

        else: 
            s2_canon, s2_transform = game1.get_canonical_key(game1.board)
            s2_canon_board = game1.key_state_to_board(s2_canon)
            a2_canon_possible = game1.get_possible_actions(s2_canon_board)

            a2_canon_chosen = agent2.choose_action(s2_canon, a2_canon_possible)

            if last_s2_canon is not None:
                agent2.update_q_table(last_s2_canon, last_a2_canon, 0.0, s2_canon, a2_canon_possible)
            
            a2_raw_chosen = game1.inverse_action(a2_canon_chosen, s2_transform)

            r2 = game1.next_move(a2_raw_chosen[0], a2_raw_chosen[1])
            
            s2_next_canon, _ = game1.get_canonical_key(game1.board)

            if game1.gameover:
                if r2 == 2.0: 
                    total_losses +=1 
                    #agent2.update_q_table(s2_canon, a2_canon_chosen, 1.0, s2_next_canon, [])
                    if last_s1_canon is not None:
                        agent1.update_q_table(last_s1_canon, last_a1_canon, -1.0, None, [])
                else:  
                    total_draws += 1
                    #agent2.update_q_table(s2_canon, a2_canon_chosen, 0.0, s2_next_canon, [])
                    if last_s1_canon is not None:
                        agent1.update_q_table(last_s1_canon, last_a1_canon, -0.2, None, [])
                break

            last_s2_canon, last_a2_canon = s2_canon, a2_canon_chosen

            current_player = 1 
            
    if (epoch + 1) % sub_count == 0:
            win_rate = total_wins / sub_count
            loss_rate = total_losses / sub_count
            draw_rate = total_draws / sub_count
            
            win_log.append(win_rate)
            loss_log.append(loss_rate)
            draw_log.append(draw_rate)
            
            print(f"Epoch {epoch+1}/{num_games} | Win Rate: {win_rate:.2f} | Loss Rate: {loss_rate:.2f} | Draw Rate: {draw_rate:.2f}")
            
            total_wins = 0
            total_losses = 0
            total_draws = 0

print("Training complete!")
plt.figure(figsize=(12, 6))
plt.plot(win_log, label='AI Win Rate')
plt.plot(loss_log, label='AI Loss Rate')
plt.plot(draw_log, label='Draw Rate')
plt.xlabel('Batch of 1000 Games')
plt.ylabel('Rate')
plt.title('AI Training Performance Over Time')
plt.legend()
plt.grid(True)
plt.show()