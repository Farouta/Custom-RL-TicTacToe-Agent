from src.environment import tictactoe
from src.agent import Agent
import random

game2= tictactoe()
agent2=Agent()

agent2.load_model("./trained_agents/trained_agent2.pkl")
agent2.epsilon = 0.0
current_player=random.randint(1,2)
game2.current_player = current_player

while (not (game2.gameover)):
    if current_player==1 :
        canon_state_key , transform =game2.get_canonical_key(game2.board)
        canon_board=game2.key_state_to_board(canon_state_key)
        canon_possible_actions=game2.get_possible_actions(canon_board)

        canon_action=agent2.choose_action(canon_state_key,canon_possible_actions)
        action=game2.inverse_action(canon_action, transform)
        game_flag=game2.next_move(action[0],action[1]) #who won or if draw
        current_player=2
    game2.view_board()

    if game2.gameover:
        if game_flag == 1.0:
            print("--- AGENT WINS ---")
        else:
            print(game2.check_win)
            print(game_flag)
            print("--- GAME IS A DRAW ---")
        break 
    if current_player==2:
        print("\nYour Turn:")
        game_flag = game2.player_move()
        current_player=1
    if game2.gameover:
        if game_flag == 2.0:
            game2.view_board()
            print("--- YOU WIN ---")
        else:
            print("--- GAME IS A DRAW ---")
        break