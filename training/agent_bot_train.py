from src.environment import tictactoe
from src.agent import Agent
import random
import matplotlib.pyplot as plt

game1=tictactoe()
agent2=Agent()

win_log = []
loss_log = []
draw_log = []
total_wins = 0
total_losses = 0
total_draws = 0

num_games = 50000
batch=1000
for epoch in range (num_games):
    game1.reset_game()
    last_agent_s = None
    last_agent_a = None
    current_player=random.randint(1,2)
    game1.current_player = current_player
    agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay_rate, agent2.epsilon_min)
    agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay_rate, agent2.epsilon_min)
    while (not (game1.gameover)):
        
        if current_player==1 :
            canon_state_key , transform =game1.get_canonical_key(game1.board)
            canon_board=game1.key_state_to_board(canon_state_key)
            canon_possible_actions=game1.get_possible_actions(canon_board)


            if last_agent_s is not None:
                agent2.update_q_table(last_agent_s, last_agent_a, 0.0, canon_state_key, canon_possible_actions)
            canon_action=agent2.choose_action(canon_state_key,canon_possible_actions)
            action=game1.inverse_action(canon_action, transform)
            game_flag=game1.next_move(action[0],action[1]) #who won or if draw
            
            canon_next_state_key, next_transform=game1.get_canonical_key(game1.board)
            canon_board=game1.key_state_to_board(canon_next_state_key)
            canon_next_possible_actions=game1.get_possible_actions(canon_board)

            last_agent_s = canon_state_key
            last_agent_a = canon_action


            if (game1.gameover):
                if game_flag == 1.0 : 
                    total_wins += 1 
                    learning_reward = 1.0
                else:
                    total_draws += 1
                    learning_reward = -0.2
                agent2.update_q_table(canon_state_key, canon_action, learning_reward, "", [])
                break
            current_player=2
        else:
            game_flag=game1.bot_move()

            if (game1.gameover):

                if  game_flag== 2.0: 
                    total_losses += 1 
                    learning_reward = -1.0
                else:
                    total_draws += 1
                    learning_reward = -0.2

                if last_agent_s is not None:
                    agent2.update_q_table(last_agent_s, last_agent_a, learning_reward, "", [])
                
                break
            current_player=1

    if (epoch + 1) % batch== 0:
            win_rate = total_wins / batch
            loss_rate = total_losses / batch
            draw_rate = total_draws / batch
            
            win_log.append(win_rate)
            loss_log.append(loss_rate)
            draw_log.append(draw_rate)
            print(f"Epoch {epoch+1}/{num_games} | Win Rate: {win_rate:.2f} | Loss Rate: {loss_rate:.2f} | Draw Rate: {draw_rate:.2f}")
            total_wins = 0
            total_losses = 0
            total_draws = 0
agent2.save_model("trained_agent2.pkl")


print("Training complete!")
plt.figure(figsize=(12, 6))
plt.plot(win_log, label='AI Win Rate')
plt.plot(loss_log, label='AI Loss Rate')
plt.plot(draw_log, label='Draw Rate')
plt.xlabel(f'Batch of {batch} Games')
plt.ylabel('Rate')
plt.title('Phase 1: Policy Convergence vs. Stochastic Baseline')
plt.legend()
plt.grid(True)
plt.show()



