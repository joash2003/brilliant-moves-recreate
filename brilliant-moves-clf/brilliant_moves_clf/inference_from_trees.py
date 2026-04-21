import os
import pickle
from datetime import datetime
import glob
import networkx as nx
import numpy as np
import traceback
import warnings
from sklearn import preprocessing
import torch
from torch import nn
import argparse

'''
This script will classify moves as brilliant or not, based on their game trees. Must run generate_trees.py first.
Must have following directory structure:

brilliant_moves_clf
|-- data_extraction.py
|-- models
|   |-- model-7936-2.pth (state_dict for the neural network)
|-- moves
|   |-- move_id_1
|   |   |-- fen.txt (from board before move. This script does not use this file, it is needed for generate_trees.py)
|   |   |-- uci.txt (the move to classify.)
|   |-- move_id_2
        .
        .
        .
|-- trees
|   |-- lc0
|   |   |-- move_id_1
|   |   |   |-- tree_0_1.gml
            .
            .
            .
|   |   |   |-- tree_0_5.gml.txt
|   |-- move_id_2
        .
        .
        .
|   |-- maia
|   |   |-- move_id_1
|   |   |   |-- tree_1_1.gml
            .
            .
            .
|   |   |   |-- tree_1_5.gml.txt
|   |-- move_id_2
        .
        .
        .
'''

class NeuralNetworkDropout(nn.Module):
    def __init__(self, h1, h2, h3, dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.tree_layer = nn.Sequential(
            nn.Linear(22*18+2, h1), #22*18+2 features per tree size per weight
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.weight_layer = nn.Sequential(
            nn.Linear((h1+22*2+2)*2, h2), #Takes in raw features and h1 
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(h2*5, h3),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h3,1),
        )

    def forward(self, x):
        x = self.flatten(x)
        h1s = [[0 for weight in range(2)] for tree in range(5)]
        for tree in range(5):
            for weight in range(2):
                tree_feats = x[:, index_flat(weight, tree):index_flat(weight, tree+1)]
                h1s[tree][weight] = self.tree_layer(tree_feats)
        h2s = [0 for tree in range(5)]
        for tree in range(5):
            feats = []
            for weight in range(2):
                feats.append(x[:, index_flat(weight, tree, 0):index_flat(weight, tree, 0)+2]) #First 2 boolean features
                feats.append(x[:, index_flat(weight, tree, subtree=0):index_flat(weight, tree, subtree=2)]) #Features for parent and move of interest subtrees
                feats.append(h1s[tree][weight])
            feats = torch.cat(feats,1)
            h2s[tree] = self.weight_layer(feats)
        return self.final_layer(torch.cat(h2s,1))

def get_subtree_data(G, uci, root):
    """
    get_subtree_data extracts data from a game subtree rooted at a specified node.
    
    :param G: the full game tree to process, as a NetworkX graph object
    :param uci: the move of interest (the move you wish to classify), in string UCI format
    :param root: the board state at which the subtree is rooted
    :return: the extracted data from the subtree
    """
    improving_moves = []
    adv_moves = []
    losing_moves = []
    disadv_moves = []
    move_node = None
    max_Q = -1
    max_P = 0
    max_N = 0
    root_Q = float(G.nodes[root]['Q'])
    try:
        root_P = float(G.nodes[root]['P'])
    except:
        root_P = 0
    root_N = float(G.nodes[root]['N'])
    move_is_best = False
    for i in G.successors(root): #For each possible explored move from the root board position
        node = G.nodes[i]
        N = int(node['N'])
        Q = float(node['Q'])
        P = float(node['P'])
        move = node['move']
        
        if N > max_N:
            max_N = N
        if Q > max_Q:
            max_Q = Q
        if P > max_P:
            max_P = P
        
        if move == uci:
            move_node = i
        else:
            if Q - (-root_Q) > 0:
                improving_moves.append(i)
            else:
                losing_moves.append(i)
            if Q > 0:
                adv_moves.append(i)
            else:
                disadv_moves.append(i)
    
    if move_node is not None:
        move_is_best = float(G.nodes[move_node]['Q']) >= max_Q
    
    try:
        descendants = nx.descendants(G,root) #All nodes reachable from the root
        bf = (len(descendants)-1)/len([x for x in descendants if G.out_degree(x)==0 and G.in_degree(x)==1]) # (N-1)/(number of leaves)
    except:
        bf = -1
    
    width = {}
    for k,v in nx.shortest_path_length(G,root).items():
        width[v] = width.get(v,0) + 1 #Number of nodes at each depth v
    try:
        height = max(k for k,v in width.items()) #Total height of subtree from root
    except:
        height = -1
    
    return improving_moves, adv_moves, losing_moves, disadv_moves, move_node if move_node is not None else None, max_Q, max_P, max_N, root_Q, root_P, root_N, bf, width, height, move_is_best

def get_data(G, uci):
    """
    get_data extracts all necessary data from a game tree.
    
    :param G: the game tree to process, as a NetworkX graph object
    :param uci: the move of interest (the move you wish to classify), in string UCI format
    :return: the extracted data from the tree
    """
    data = []
    
    # Extracts the features for the tree rooted at the node before the move of interest is made.
    parent_data = get_subtree_data(G, uci, 0)
    data.append(parent_data)
    
    # If the move of interest is not in the tree, leave feature as None, else extract the features for the tree rooted at the node after the move.
    if parent_data[4] is None:
        data.append(None)
    else:
        data.append(get_subtree_data(G, uci, parent_data[4]))
        
    # For each of the 4 move subsets (improving_moves, adv_moves, losing_moves, disadv_moves), extract features for each tree rooted at the node after each move.
    for subset in parent_data[:4]:
        subset_data = []
        for i in subset:
            subset_data.append(get_subtree_data(G, uci, i))
        data.append(subset_data)

    # data = [parent, move of interest, improving_moves, adv_moves, losing_moves, disadv_moves]
    return data

def feature_transform(subtree_data):
    """
    feature_transform turns extracted data from a subtree into aggregated features
    
    :param subset: data corresponding to a subtree from the full game tree (can be the full tree)
    :return: the features representing this subtree
    """
    features = []
    features.extend([len(i) for i in subtree_data[:4]])#Length of each move type subset
    features.extend(subtree_data[5:12])#Single value features
    widths = list(subtree_data[12].values())
    widths += [0] * (8-len(widths))
    features.extend((widths[1:8]))#width of first 7 layers
    features.append(np.mean(widths))#mean
    features.append(np.std(widths))#std
    features.append(np.max(widths))#max
    features.append(subtree_data[13])#height
    return features

def index_flat(weight, tree, index = 0, subtree = None, agg = 0):
    """
    index_flat allows you to access specific features from the total move feature vector of size 3980
    
    :param weight: the weight index of the feature you are trying to access (default - 0: lc0, 1: maia)
    :param tree: the tree size index of the feature you are trying to access (default: 0: 10^1 size, 1: 10^2 ... 4: 10^5)
    :param index: the index of the feature.
                  If no subtree is provided, then (0: Is move of interest in tree? 1: Is move the best move?)
                  If subtree is provided, index matches feature_transform, as shown below:
                    Index  :        0  ,  1 , 2     , 3           ,     4   5   6,      7   8   9,  10,          11 - 17, 18     19    20       , 21
                    Feature: Num of imp, adv, losing, disadv moves, max q & p & n, root q & p & n,  bf, width for 1 - 8 , mean & std & max width, height
    :param subtree: the index of the subtree containing the feature. (None: See above, 0: Parent, 1: Move of Interest, 2-5: imp, adv, losing, disadv moves)
    :param agg: the index of the aggregation used for the subtree. (0: mean, 1: std, 2: max, 3: min)
    :return: the index of the feature within the total move feature vector of size 3980
    """
    base = weight * 5*(22*18+2) + tree * (22*18+2)
    if subtree is None:
        return base + index
    elif subtree < 2:
        return base + 2 + 22*subtree + index
    else:
        subset = subtree - 2
        return base + 2 + 22*2 + 22*4*subset + agg*22 + index

def parse_trees(moves_dir = 'moves'):
    """
    parse_trees extracts data from the trees for each move, and returns it as numpy array
    
    :param moves_dir: path to directory containing the move folders, each containing the FEN for the board and UCI for the move
    :return: a 2D numpy array with shape (N, 3980) containing all the features for all the moves 
    """
    weights = ['lc0','maia'] #the name of the weights types, not the paths
    dirs = [i[0] for i in os.walk(moves_dir)][1:]
    X = []

    #lens, max q, max p/n, root q, root p/n+bf, width for 1-8 + 3 stats, height
    defaults = (0,)*4 + (-1,) + (0,)*2 + (-1,) + (0,)*3 + (0,)*10 + (0,)

    for move_num, move in enumerate(dirs):
        move_name = os.path.basename(move)
        X.append([])
        with open(os.path.join(move,'uci.txt'),'r') as f:
            uci = f.read()
        for wi, weight in enumerate(weights):
            trees_dir = os.path.join('trees',weight,move_name)
            for tree_num in range(1,6):
                #if len(data[t][board_num][wi][tree_num-1]) == 0:
                print("Processing:",move_name,weight,tree_num,datetime.now())
                tree_dir = glob.glob(os.path.join(trees_dir,f"tree_{wi}_{tree_num}.gml"))
                if len(tree_dir) != 1:
                    print(f"No file or duplicate files found for {os.path.join(trees_dir,f'tree_{wi}_{tree_num}.gml')}")
                try:
                    G = nx.read_gml(tree_dir[0], label=None)
                    #data[board_num][wi][tree_num-1] = get_data(G, uci)
                    tree_data = get_data(G, uci)
                    
                    features = []
                    move_in_tree = tree_data[0][4] is not None
                    features.append(move_in_tree) #Is move of interest in tree?
                    features.append(tree_data[0][-1]) #Is move the best move?
                    features.append(feature_transform(tree_data[0])) #Add features for tree rooted at parent board
                    if move_in_tree:
                        features.append(feature_transform(tree_data[1])) #Add features for subtree rooted at move of interest
                    else:
                        features.append(defaults) #Add default blank values if move not in tree
                    for s_num, subset in enumerate(tree_data[2:]): #For each of the 4 sets of possible moves
                        if len(subset)==0:
                            for i in range(4):
                                features.append(defaults) #If the subset contains no moves, use blank defaults
                        else:
                            # Aggregate the features across all trees within the subset of moves
                            subset_features = np.array([feature_transform(i) for i in subset])
                            features.append(np.mean(subset_features, axis=0))
                            features.append(np.std(subset_features, axis=0))
                            features.append(np.max(subset_features, axis=0))
                            features.append(np.min(subset_features, axis=0))
                    X[-1].append(features[0])
                    X[-1].append(features[1])
                    for subtree_features in features[2:]:
                        X[-1].extend(subtree_features)
                except Exception as e:
                    print(traceback.format_exc())
                    print("Bad tree:",move_name,weight,tree_num)
                    continue
                
    X = np.array(X)
    return X


def apply_scaler(X, scaler_path=None):
    """
    Normalizes feature matrix X using a persisted StandardScaler. If no scaler
    path is provided or the file is missing, we fall back to fitting on X
    itself (the original author's behavior) but emit a warning -- this fallback
    makes results batch-dependent and should only be used when no training
    scaler is available. Fix for the StandardScaler bug documented in REPORT.md.
    """
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[scaler] loaded pre-fit scaler from {scaler_path}")
        return scaler.transform(X)
    warnings.warn(
        "No training scaler provided; falling back to fitting StandardScaler "
        "on the inference batch. Results will depend on batch composition. "
        "See REPORT.md 'Bug 3' for details.",
        RuntimeWarning,
    )
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)

def run_inference(X, state_dict = os.path.join('models','model-7936-2.pth'), moves_dir = 'moves'):
    """
    run_inference will print the classifier confidence score and prediction for moves given in X.
    
    :param X: the numpy array containing all the data, in the shape (N, D) for N samples and D features (D=3980).
    :param state_dict: path to the state dict file storing the weights of the neural network.
    :return: None, the output is printed to stdout.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    h1, h2, h3 = 25, 400, 50 #the hidden layer size hyperparameters in the network
    network = NeuralNetworkDropout(h1,h2,h3,dropout=0.2).to(device)
    network.load_state_dict(torch.load(state_dict, map_location=device))
    network.eval()

    X = torch.tensor(X)

    with torch.no_grad():
        outputs = network(X.to(device).float())
        
    dirs = [i[0] for i in os.walk(moves_dir)][1:]
    for move_num, move in enumerate(dirs):
        move_name = os.path.basename(move)
        # Need to reverse the output to match conventions, because in my training, 0 was brilliant and 1 was non-brilliant.
        print(f'{move_name}: {torch.sigmoid(-outputs[move_num]).item():.2f}, {"Brilliant" if outputs[move_num].item() < 0 else "Not Brilliant"}.')
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'This script will use the neural network to predict if the given moves are brilliant or not')
    parser.add_argument("-d", "--moves_dir", help = "Path to directory containing the move folders.", default = 'moves')
    parser.add_argument("-s", "--state_dict", help = "Path to the state dict file storing the weights of the neural network.", default = os.path.join('models','model-7936-2.pth'))
    parser.add_argument("--scaler", help = "Path to a pickled sklearn StandardScaler fit on the training set. If omitted, scaler is fit on the inference batch (original buggy behavior).", default=os.path.join('models','scaler.pkl'))
    args = parser.parse_args()
    
    X = parse_trees(moves_dir=args.moves_dir)
    X = apply_scaler(X, scaler_path=args.scaler)
    run_inference(X, state_dict=args.state_dict, moves_dir=args.moves_dir)