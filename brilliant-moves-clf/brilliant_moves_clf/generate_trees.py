from subprocess import Popen, PIPE, DEVNULL
import os
import time
import shutil
import argparse

'''
This script will convert moves (in FEN format) to game trees using multiple weights (lc0 and maia) and multiple sizes (10**1 to 10**5)
Must have following directory structure:

brilliant_moves_clf
|-- generate_trees.py
|-- moves
|   |-- move_id_1
|   |   |-- fen.txt (from board before move)
|   |   |-- uci.txt (the move to classify. This script does not use this file yet, but is needed for next steps)
|   |-- move_id_2
        .
        .
        .
|-- weights (to change the weights, you need to modify the names in this script)
|   |-- 768x15x24h-t82-swa-7464000.pb.gz 
|   |-- maia-1900.pb.gz
|-- lc0 (code from github, built. Original script uses 0.30.0. Source: https://github.com/LeelaChessZero/lc0)
|   |-- build
        .
        .
        .
        
Resulting trees will be placed in the trees folder, in subfolder:
    lc0, if generated with the lc0/selfplay weights,
    maia, if generated with maia weights.
Each move will have a folder containing all the tree of all sizes.
'''

def generate_trees(weight_files = [os.path.join("weights", i) for i in ['768x15x24h-t82-swa-7464000.pb.gz', 'maia-1900.pb.gz']], moves_dir = 'moves', attempts_per_move = 2, reset_limit = 4, timeout = 600):
    """
    generate_trees will convert moves (in FEN format) to game trees using multiple weights (lc0 and maia) and multiple sizes (10**1 to 10**5)
    
    :param weight_files: Paths of lc0 weight files. Ordered so self-play weights are 0 index, maia is 1 index
    :param moves_dir: Path to directory containing the move folders, each containing the FEN for the board and UCI for the move
    :param attempts_per_move: Adjusts how many times to retry generating a move if it fails. Recommended >= 2
    :param reset_limit: Number of moves that are processed before restarting Lc0. I believe Lc0 maintains a cache which slows down computation. Resetting every 4 moves seemed to work well, adjust according to your resources
    :param timeout: Set max time allowed for Lc0 to generate and write a tree, will fail if this time is exceeded (will retry based on attempts_per_move)
    :return: None, the trees are written directly to the trees folder
    """
    dirs = [i[0] for i in os.walk(moves_dir)][1:]

    # If tree.gml exists, it blocks a new one from being generated.
    try:
        os.remove('tree.gml')
    except OSError:
        pass

    for w, weights in enumerate(weight_files):
        lc0_settings = [os.path.join('lc0','build','release','lc0'),'--backend=cuda-auto',f'--weights={weights}']
        p = Popen(lc0_settings, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
        
        dest_dir = os.path.join('trees',f'{"lc0" if w == 0 else "maia"}')
        os.makedirs(dest_dir,exist_ok=True)
        
        move_count = 0
        for move in dirs:
            try:
                move_name = os.path.basename(move)
                print(f"Processing using weights {w}, move: {move_name}")
                
                move_count += 1
                if move_count % reset_limit == 0:
                    p.kill()
                    p = Popen(lc0_settings, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
                    move_count = 0
                    
                with open(os.path.join(move,'fen.txt'), 'r') as f:
                    fen = f.read()
                    
                for nodes in range(1,6): # Generate trees with 10**1 to 10**5 nodes
                    try:
                        out_path = os.path.join(dest_dir,move_name,f'tree_{w}_{nodes}.gml')
                        if not(os.path.isfile(out_path)): # Prevents overwrites
                            for tries in range(attempts_per_move): # Can fail if tree doesn't finish writing by the timeout period, or other reasons.
                                # Write instructions to Lc0
                                p.stdin.write(str.encode(f'ucinewgame\n'))
                                p.stdin.write(str.encode(f'position fen {fen}\n'))
                                p.stdin.write(str.encode(f'go nodes {int(10**nodes)}\n'))
                                p.stdin.flush()
                                
                                # Succeeds when tree file finishes writing within timeout period
                                timer = 0
                                sleep_time = 0.2
                                success = False
                                prev_size = -1
                                while(timer < timeout):
                                    time.sleep(sleep_time)
                                    timer += sleep_time
                                    if os.path.isfile('tree.gml'):
                                        curr_size = os.path.getsize('tree.gml')
                                        if curr_size == prev_size:
                                            success = True
                                            break
                                        else:
                                            prev_size = curr_size
                                            
                                if success:
                                    try:
                                        os.mkdir(os.path.join(dest_dir,move_name))
                                    except:
                                        pass
                                    shutil.move('tree.gml', out_path)
                                    break
                                else:
                                    p.kill()
                                    p = Popen(lc0_settings, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
                                    move_count = 0
                            if not(success):
                                print(f"Timeout with {move} @ nodes {nodes}")
                                break
                    except Exception as e:
                        print(f"Error with {move} @ nodes {nodes}: {e}")
            except Exception as e:
                print(f"Error with {move}: {e}")
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'This script will convert moves (in FEN format) to game trees using multiple weights (lc0 and maia) and multiple sizes (10**1 to 10**5)')
    parser.add_argument("-s", "--selfplay_weights", help = "Name of selfplay lc0 weight file.", default = os.path.join("weights", '768x15x24h-t82-swa-7464000.pb.gz'))
    parser.add_argument("-m", "--maia_weights", help = "Name of maia weight file.", default = os.path.join("weights", 'maia-1900.pb.gz'))
    parser.add_argument("-d", "--moves_dir", help = "Path to directory containing the move folders.", default = 'moves')
    parser.add_argument("-a", "--attempts_per_move", help = "Adjusts how many times to retry generating a move if it fails.", type=int, default = 2)
    parser.add_argument("-r", "--reset_limit", help = "Number of moves that are processed before restarting Lc0.", type=int, default = 4)
    parser.add_argument("-t", "--timeout", help = "Set max time allowed for Lc0 to generate and write a tree.",type=float, default = 600)
    
    args = parser.parse_args()
    generate_trees([args.selfplay_weights, args.maia_weights], args.moves_dir, args.attempts_per_move, args.reset_limit, args.timeout)