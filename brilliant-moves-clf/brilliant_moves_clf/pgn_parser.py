import chess
import chess.pgn
import os
import io
import argparse

def moves_from_pgn(pgn_str, output_dir = 'moves', variations = True, split = False):
    pgn_io = io.StringIO(pgn_str)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None: #Note, pgn format supports concatenating multiple games within one string.
            break
        
        exporter = chess.pgn.StringExporter()
        game_pgn_str = game.accept(exporter)
        try:
            white = game.headers["White"]
            white = white if white != "?" else "Blank"
            black = game.headers["Black"]
            black = black if black != "?" else "Blank"
            game_name = white + "_vs_" + black + "_"
        except:
            game_name = ''
        game_name += hex(hash(game_pgn_str) % 0xffffff)[2:]
        
        moves = []
        queue = [game]
        while queue:
            curr = queue.pop(0)
            if variations:
                next_moves = curr.variations
            else:
                next_moves = [curr.next()]
            for v in next_moves:
                queue.append(v)
                moves.append(v)

        for i, move in enumerate(moves):
            fen = move.parent.board().fen()
            uci = move.uci()
            
            if split:
                directory = os.path.join(output_dir, 'white' if i%2==0 else 'black', f'{game_name}_{i}_{uci}')
            else:
                directory = os.path.join(output_dir,f'{game_name}_{i}_{uci}')
            os.makedirs(directory,exist_ok=True)
            
            with open(os.path.join(directory,'fen.txt'),'w') as f:
                f.write(fen)
            with open(os.path.join(directory,'uci.txt'),'w') as f:
                f.write(uci)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'This script will extract moves from a pgn, and store them in the correct format.')
    parser.add_argument('pgn_str', help = "Path to output directory that will contain the move folders.")
    parser.add_argument("-d", "--moves_dir", help = "Path to output directory that will contain the move folders.", default = 'moves')
    parser.add_argument('--variations', action=argparse.BooleanOptionalAction, help = "Enable variation moves to be parsed.", default = True)
    parser.add_argument("--file", action=argparse.BooleanOptionalAction, help = "pgn_str is a path to a pgn file", default = False)
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, help = "If true, split the white and black moves into different subdirectories", default = False)
        
    args = parser.parse_args()
    pgn = args.pgn_str
    if args.file:
        with open(args.pgn_str, 'r') as f:
            pgn = f.read()
    moves_from_pgn(pgn, args.moves_dir, args.variations, args.split)
    
#response = requests.get(f'https://lichess.org/api/study/{code}.pgn')
