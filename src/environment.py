import torch
import random
import numpy as np

class tictactoe :
    def __init__(self):
        self.board=torch.zeros([3,3],dtype=torch.int8)
        self.current_player = 1
        self.current_turn= 0
        self.gameover=False

    def view_board(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("\n    0   1   2")
        for i in range(3):
            row = [symbols[self.board[i, j].item()] for j in range(3)]
            print(f"{i}   {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("   ---+---+---")


    def check_win(self, row_in , col_in):

        def checking (current):
            if (torch.all(current==1).item() ):
                self.gameover=True
                return 1.0
            elif(torch.all(current==2).item()):
                self.gameover=True
                return 2.0
            return False  


        col=self.board[:,col_in]
        row=self.board[row_in, :]
        result=checking(row)
        if result:
            return result
        result=checking(col)
        if result:
            return result
        
        if row_in == col_in:
            diag1 = torch.diag(self.board)
            result = checking(diag1)
            if result:
                return result

        if row_in + col_in == self.board.size(0) - 1:
            diag2 = torch.diag(torch.fliplr(self.board))
            result = checking(diag2)
            if result:
                return result
        if(self.current_turn ==9) :
            self.gameover=True
            
        return 0.0
        


    def next_move(self,row,col):
        if (self.gameover==False):
            if not(0 <= row <= 2 and 0 <= col <= 2):
                print("Move is outside the board (must be between 0 and 2).")
                return False
            elif (self.board[row][col]!=0 ):
                print("move is invalid")
                return False
            else:
                self.board[row][col]=self.current_player
                self.current_turn +=1
                self.current_player +=1
                if (self.current_player==3) :
                    self.current_player=1
                return self.check_win(row,col)

        else:
            print("Game is over no more turns possible")
    def bot_move(self):
        random_move=np.random.randint(low=0,high=9-self.current_turn)
        free_slots=torch.nonzero(self.board==0)
        i=free_slots[random_move][0]
        j=free_slots[random_move][1]
        return self.next_move(i,j)
    

    def get_possible_actions(self, board):
        free_slots=torch.nonzero(board==0).tolist()
        return [tuple(row) for row in free_slots]
    

    def player_move(self):
        r=int(input("Insert Row= "))
        c=int(input("Insert Col= "))
        return self.next_move(r,c)

    
    def play_game(self):
        random_start=np.random.randint(low=0,high=2)==0

        if (random_start==0):
            print("Bot starts first")
            self.bot_move()
        else:
            print("player starts first")
            self.player_move()
        self.view_board()

        while(not(self.gameover)):
            if (random_start==0):
                self.player_move()
                self.bot_move()
            if (random_start==1):
                self.bot_move()
                self.player_move()
            self.view_board()
    
    def get_state_key(self,board):
        array=board.flatten().numpy()
        return "".join(map(str,array))                      
    def reset_game(self):
        self.board=torch.zeros([3,3],dtype=torch.int8)
        self.current_player=1
        self.current_turn=0
        self.gameover=False
    def get_canonical_key(self,board):
        
        all_states={}
        all_states[self.get_state_key(board)]="00"

        for i in range (1,4):
            rotated_board=torch.rot90(board ,k=i)
            all_states[self.get_state_key(rotated_board)]="0"+str(i)

        all_states[self.get_state_key(torch.fliplr(board))]="10"    

        for i in range (1,4):
            rotated_board=torch.rot90(torch.fliplr(board),k=i)
            all_states[self.get_state_key(rotated_board)]="1"+str(i)

        canon_key = sorted(all_states)[0]
        transform=all_states[canon_key]
        return canon_key , transform



    def inverse_action(self, action, transform):
        def mirror(action):
            if action[1]==1:
                return action
            elif action[1]==0:
                return (action[0],2)
            elif action[1]==2:
                return (action[0],0)
        k=int(transform[1])

        if action==(1,1) :
            return action
        corner=[(0,0),(2,0),(2,2),(0,2)]
        side=[(0,1),(1,0),(2,1),(1,2)]
        if action in corner :
            index=corner.index(action)
            new_index=(index-k)%len(corner)
            action= corner[new_index]
        else :
            index=side.index(action)
            new_index=(index-k)%len(side)
            action= side[new_index]
        if transform[0]=="1":
            action=mirror(action)
        return action


        
    def key_state_to_board(self,key_state):
        counter=0
        board=torch.zeros(3,3, dtype=torch.int8)
        for i in range (3):
            for j in range (3):
                board[i][j]=int(key_state[counter])
                counter +=1
        return board
        
