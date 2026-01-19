# /usr/bin/python
# Training script for GAT-ETM with MIMIC-III dataset
# Adapted from main_getm.py to work with MIMIC-III public dataset
#
# Key differences from main_getm.py:
# 1. Automatically loads vocab info from KG_EMBED/embed_augmented/vocab_info.pkl
# 2. Automatically detects and loads graph/embedding files from KG_EMBED/embed_augmented/
# 3. Automatically creates metadata.txt if it doesn't exist
# 4. Uses code types and vocab sizes from KG vocab info
#
# Prerequisites:
# - KG embeddings must be generated first using build_kg_mimic.py
# - Data files (bow_train.npy, bow_test.npy, etc.) must be prepared in data/ directory
#
# Usage:
#   python main_getm_mimic.py --data_path data/ --kg_embed_dir KG_EMBED/embed_augmented --epochs 50

from __future__ import print_function

import argparse
import torch
import numpy as np
import os, time
import math
import ast
from tqdm import tqdm
import pickle
import networkx as nx
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity
from dataset import normalize, csc2tensor
from dataset import NontemporalDataset 

from IPython import embed 

parser = argparse.ArgumentParser(description='The Embedded Topic Model - MIMIC-III')

parser.add_argument('--data_path', type=str, default='data/', help='directory containing data')
parser.add_argument('--meta_file', type=str, default='metadata', help='file containing meta data under data_path (without .txt extension)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--kg_embed_dir', type=str, default='KG_EMBED/embed_augmented', help='directory containing KG embeddings')

parser.add_argument('--save_path', type=str, default='results/', help='path to save results')

parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')

### model-related arguments
parser.add_argument('--add_freq', type=int, default=0, help='whether to consider baseline frequency or not')

parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=256, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=256, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=128, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')
parser.add_argument('--nlayer', type=int, default=3, help='number of layers for theta')
parser.add_argument('--upper', type=int, default=10, help='upper boundary for Gaussian variance')
parser.add_argument('--lower', type=int, default=-10, help='lower boundary for Gaussian variance')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train...150 for 20ng 100 for others')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')

parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')

parser.add_argument('--nonmono', type=int, default=5, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=bool, default=True, help='normalize the bows or not')
parser.add_argument('--gpu_device', type=str, default="cuda", help='gpu device name')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=1, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=512, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tq', action='store_true', help='whether to compute topic coherence or not')
parser.add_argument('--sharea', action='store_true', help='')

parser.add_argument('--drug_imputation', action='store_true', help='')
parser.add_argument('--dc_thr', type=int, default=5)
parser.add_argument('--impute_k', type=int, default=5, help='')
parser.add_argument('--loss', type=str)
parser.add_argument('--gamma', type=float)

args = parser.parse_args()

device = torch.device('cuda:'+args.gpu_device if torch.cuda.is_available() else "cpu")
args.device = device

# Load vocab info from KG embeddings
print("Loading vocab info from KG embeddings...")
vocab_info_path = os.path.join(args.kg_embed_dir, 'vocab_info.pkl')
if not os.path.exists(vocab_info_path):
    raise FileNotFoundError(f"Vocab info not found at {vocab_info_path}. Please run KG builder first.")

with open(vocab_info_path, 'rb') as f:
    vocab_info = pickle.load(f)

print(f"Available vocab types in KG: {list(vocab_info.keys())}")

# Determine code types - use 'icd' and 'atc' as default (can be customized)
# Filter out empty vocabs
available_code_types = [ct for ct in ['icd', 'atc'] if ct in vocab_info and len(vocab_info[ct]) > 0]
if len(available_code_types) == 0:
    # Fallback: use all available code types
    available_code_types = [ct for ct in vocab_info.keys() if len(vocab_info[ct]) > 0]

args.code_types = available_code_types
vocab_size = [len(vocab_info[ct]) for ct in args.code_types]
args.vocab_size = vocab_size
args.vocab_cum = np.cumsum([0]+args.vocab_size)

print(f"Using code types: {args.code_types}")
print(f"Vocab sizes: {args.vocab_size}")
print(f"Vocab cumulative: {args.vocab_cum}")

# Load or create metadata file
metadata_path = os.path.join(args.data_path, args.meta_file + '.txt')
if os.path.exists(metadata_path):
    print(f"Loading metadata from {metadata_path}...")
    try:
        # Read metadata file line by line and parse properly
        with open(metadata_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) >= 4:
            # Parse each line - handle both string representations and actual lists
            try:
                args.code_types = ast.literal_eval(lines[0])
            except:
                # Fallback: try to parse manually if it's malformed
                code_types_str = lines[0].strip("[]\"'")
                args.code_types = [ct.strip(" \"'") for ct in code_types_str.split(',')]
            
            try:
                vocab_size = ast.literal_eval(lines[1])
                args.vocab_size = [int(_) for _ in vocab_size] if isinstance(vocab_size, list) else [int(vocab_size)]
            except:
                # Fallback: try to parse manually
                vocab_str = lines[1].strip("[]")
                args.vocab_size = [int(v.strip()) for v in vocab_str.split(',')]
            
            args.vocab_cum = np.cumsum([0]+args.vocab_size)
            
            try:
                train_embeddings = ast.literal_eval(lines[2])
                args.train_embeddings = [int(_) for _ in train_embeddings] if isinstance(train_embeddings, list) else [int(train_embeddings)]
            except:
                train_str = lines[2].strip("[]")
                args.train_embeddings = [int(v.strip()) for v in train_str.split(',')]
            
            try:
                args.embedding = ast.literal_eval(lines[3])
                if isinstance(args.embedding, str):
                    args.embedding = [args.embedding]
            except:
                embed_str = lines[3].strip("[]\"'")
                args.embedding = [e.strip(" \"'") for e in embed_str.split(',')]
            
            print("Metadata loaded successfully. Overriding code_types and vocab_size from metadata.")
        else:
            print("Metadata file exists but incomplete. Using defaults from KG vocab info.")
            args.train_embeddings = [1] * len(args.code_types)
            args.embedding = ['*'] * len(args.code_types)
    except Exception as e:
        print(f"Error loading metadata: {e}. Using defaults from KG vocab info.")
        import traceback
        traceback.print_exc()
        args.train_embeddings = [1] * len(args.code_types)
        args.embedding = ['*'] * len(args.code_types)
else:
    print(f"Metadata file not found at {metadata_path}. Using defaults from KG vocab info.")
    # Default: train embeddings from graph
    args.train_embeddings = [1] * len(args.code_types)
    args.embedding = ['*'] * len(args.code_types)
    # Optionally create metadata file
    print("Creating metadata file...")
    os.makedirs(args.data_path, exist_ok=True)
    with open(metadata_path, 'w') as f:
        f.write(str(args.code_types) + '\n')
        f.write(str(args.vocab_size) + '\n')
        f.write(str(args.train_embeddings) + '\n')
        f.write(str(args.embedding) + '\n')
    print(f"Metadata saved to {metadata_path}")

# Load graph and embeddings from KG
print(f"\nLoading graph and embeddings from {args.kg_embed_dir}...")

# Find graph and embed files
# Exclude vocab mapping files: graphnode_vocab.pkl, vocab_info.pkl
# But allow files with 'vocab' in the middle (e.g., icdatc_graph_*_renumbered_by_vocab.pkl)
vocab_mapping_files = {'graphnode_vocab.pkl', 'vocab_info.pkl'}
graph_files = [f for f in os.listdir(args.kg_embed_dir) 
               if 'graph' in f and f.endswith('.pkl') and f not in vocab_mapping_files]
embed_files = [f for f in os.listdir(args.kg_embed_dir) 
               if 'embed' in f and f.endswith('.pkl') and f not in vocab_mapping_files]

if len(graph_files) == 0 or len(embed_files) == 0:
    raise FileNotFoundError(f"Graph or embed files not found in {args.kg_embed_dir}")

# Use augmented files if available, otherwise use any available
graph_file = None
embed_file = None

for f in graph_files:
    if 'augmented' in f:
        graph_file = f
        break
if graph_file is None:
    graph_file = graph_files[0]

for f in embed_files:
    if 'augmented' in f:
        embed_file = f
        break
if embed_file is None:
    embed_file = embed_files[0]

args.graph_path = os.path.join(args.kg_embed_dir, graph_file)
args.graph_embed_path = os.path.join(args.kg_embed_dir, embed_file)

print(f"Graph file: {args.graph_path}")
print(f"Embed file: {args.graph_embed_path}")

args.graph_embed = pickle.load(open(args.graph_embed_path, 'rb'))
args.graph = pickle.load(open(args.graph_path, 'rb'))

# Check if graph is NetworkX graph or dict, and handle accordingly
if isinstance(args.graph, dict):
    raise TypeError(f"Graph file {args.graph_path} contains a dict, not a NetworkX graph. "
                   f"Please check that you're loading the correct graph file (not graphnode_vocab.pkl).")
elif isinstance(args.graph, nx.Graph):
    print(f"Graph loaded: {args.graph.number_of_nodes()} nodes, {args.graph.number_of_edges()} edges")
else:
    # Try to check if it has nodes/edges methods
    if hasattr(args.graph, 'nodes') and hasattr(args.graph, 'edges'):
        try:
            num_nodes = len(list(args.graph.nodes()))
            num_edges = len(list(args.graph.edges()))
            print(f"Graph loaded: {num_nodes} nodes, {num_edges} edges")
        except:
            raise TypeError(f"Graph object from {args.graph_path} is not a valid NetworkX graph object.")
    else:
        raise TypeError(f"Graph object from {args.graph_path} is not a NetworkX graph object. "
                       f"Type: {type(args.graph)}")

print(f"Embeddings shape: {args.graph_embed.shape}")

# Set rho_size from embeddings
args.rho_size = args.graph_embed.shape[1]
print(f"Rho size set to: {args.rho_size}")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

embeddings = {}
for i, c in enumerate(args.code_types):
    if args.embedding[i] == '*': 
        embeddings[c] = None
    else:
        embed_file = os.path.join(args.data_path, args.embedding[i])
        embeddings[c] = torch.from_numpy(np.load(embed_file)).to(device)

constraint = {}

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.drug_imputation:
    from graph_etm_DI import GETM
else:
    from graph_etm import GETM

## define model and optimizer
model = GETM(args.device, args.num_topics, args.code_types, args.vocab_size,
        args.t_hidden_size, args.rho_size, args.emb_size, args.theta_act, 
        args.graph, args.graph_embed, 
        embeddings, args.train_embeddings,
        args.enc_drop, args.upper, args.lower, args.sharea).to(device)

if args.mode == 'eval':
    ckpt = args.load_from
    with open(ckpt, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
        model.device = device
elif args.mode == 'finetune': 
    ckpt = args.load_from
    with open(ckpt, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
        model.device = device
    ckpt = os.path.join(args.save_path,
                        'getm_MIMIC_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}.mdl'.format(
                            args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
                            args.lr, args.batch_size, args.rho_size, args.train_embeddings))
    args.lr *= 0.1
    args.mode = 'train'
else:
    ckpt = os.path.join(args.save_path,
                        'getm_MIMIC_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}.mdl'.format(
                            args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
                            args.lr, args.batch_size, args.rho_size, args.train_embeddings))
    del args.graph

print('model: {}'.format(model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Custom collate function to handle sparse tensors
def sparse_collate_fn(batch):
    """
    Custom collate function to handle sparse tensors.
    Converts sparse tensors to dense before batching.
    """
    samples, indices = zip(*batch)
    
    # Convert sparse tensors to dense
    collated_samples = {}
    for key in samples[0].keys():
        tensors = []
        for s in samples:
            tensor = s[key]
            # Convert sparse to dense if needed
            if isinstance(tensor, torch.Tensor) and tensor.is_sparse:
                tensor = tensor.to_dense()
            # Ensure tensor is on correct device and has correct dtype
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.float()
            # Squeeze any extra dimensions if present (e.g., [1, vocab_size] -> [vocab_size])
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            tensors.append(tensor)
        
        # Stack tensors
        collated_samples[key] = torch.stack(tensors, dim=0)
    
    indices = torch.tensor(indices, dtype=torch.long)
    return collated_samples, indices

# Dataset
code_type_info = (args.code_types, args.vocab_size, args.vocab_cum)
train_filename = os.path.join(args.data_path, "bow_train.npy")
TrainDataset = NontemporalDataset('train', train_filename, code_type_info, device=device, drug_imputation=args.drug_imputation, drug_count_thr=args.dc_thr)
TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                collate_fn=sparse_collate_fn)

num_batches = int(np.ceil(TrainDataset.__len__()/args.batch_size))
args.log_interval = 20 # int(max(num_batches/20,np.sqrt(num_batches)))

test_filename = os.path.join(args.data_path, "bow_test.npy")
test_file_1 = os.path.join(args.data_path, "bow_test_1.npy")
test_file_2 = os.path.join(args.data_path, "bow_test_2.npy")
TestDataset = NontemporalDataset('test', (test_filename, test_file_1, test_file_2), code_type_info, device=device, drug_imputation=args.drug_imputation, drug_count_thr=args.dc_thr)
TestDataloader = DataLoader(TestDataset, batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=sparse_collate_fn)
print('Data Prepared: train:  {}, test: {}'.format(len(TrainDataset), len(TestDataset)))
print('normalization: ', args.bow_norm)

def calc_tq(m):
    m.eval()
    with torch.no_grad():
        beta = m.get_beta()
    m.train()

    train_data = TrainDataset.stack_bow

    TQ = TC = TD = 0
    TC = {}
    for i, c in enumerate(args.code_types):
        beta_i = beta[c].data.cpu().numpy()
        train_data_i = train_data[c]
        TC[c] = get_topic_coherence(beta_i, train_data_i, 3)
        TC[c] = round(TC[c], 3)

    print(f'Topic Coherence (top 3 / type) is: {TC}, mean: {np.mean(list(TC.values()))}')

    TD = {}
    for i, c in enumerate(args.code_types):
        TD[c] = get_topic_diversity(beta[c].data.cpu().numpy(), 3)
        TD[c] = round(TD[c], 3)
    print(f'Topic Diversity (top 3 / type) is: {TD}, mean: {np.mean(list(TD.values()))}')

    TQ = np.array(list(TD.values())) * np.array(list(TC.values()))
    print(f'Topic Quality is: {TQ}, mean: {np.mean(TQ)}')
    print('#' * 100)

               
def train(epoch):
    epoch_start_time = time.time()
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0

    drug_impu_loss = 0

    cnt = 0

    train_bar = tqdm(TrainDataloader, total=len(TrainDataloader), desc=f'Epoch {epoch}', position=0)
    train_log = tqdm(total=0, bar_format='{desc}', position=1)

    for idx, (sample_batch, index) in enumerate(TrainDataloader):
        optimizer.zero_grad()
        model.zero_grad()
        # Data is already dense and batched from collate_fn
        data_batch = sample_batch['Data'].to(device).float()
        # Remove extra dimension if present (from old sparse tensor format)
        if data_batch.dim() > 2:
            data_batch = data_batch.squeeze(1)

        normalized_data_batch = normalize(data_batch, args.vocab_cum, args.bow_norm)
        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta 

        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += recon_loss.item()
        acc_kl_theta_loss += kld_theta.item()

        cnt += 1

        lr = optimizer.param_groups[0]['lr']
        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)

        cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        if idx % args.log_interval == 0 and idx > 0:
            train_bar.update(args.log_interval)
            train_log.set_description_str(f'KL_theta: {cur_kl_theta} . Rec_loss: {cur_loss} . NELBO: {cur_real_loss}')

    train_bar.close()
    train_log.close()

    cur_loss = round(acc_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)

    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('Epoch----->{} .time: {:.2f} s. LR: {} .. KL_theta: {} .. Rec_loss: {} ... NELBO: {}'.
          format(epoch, time.time()-epoch_start_time, optimizer.param_groups[0]['lr'], cur_kl_theta,
                               cur_loss, cur_real_loss))
    print('*' * 80)

class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, weights, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = weights.clone().detach().to(device)
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCELoss(reduction='none')(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data)
        pt = torch.exp(-BCE_loss)
        w =  at * (1-pt)**self.gamma
        F_loss = w / (w.sum(0)+1e-9) * BCE_loss
        return F_loss.sum(1).mean()

class WeightedKLDivLoss(torch.nn.Module):
    def __init__(self, weights, gamma=0, weighted_by_sample=False):
        super(WeightedKLDivLoss, self).__init__()
        self.loss_func = nn.KLDivLoss(reduction='none')
        self.weights_of_codes = weights
        self.gamma = gamma
        self.weighted_by_sample = weighted_by_sample
    
    def forward(self, log_prob, targets):
        if self.weighted_by_sample:
            targets, unnorm_targets = targets
        loss = self.loss_func(log_prob, targets)
        weighted_loss = loss * self.weights_of_codes**self.gamma
        if self.weighted_by_sample:
            weighted_loss *= unnorm_targets.sum(1).unsqueeze(1)

        return weighted_loss.sum(1).mean()

def train_DI(epoch):
    epoch_start_time = time.time()
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    acc_drug_impu_loss = 0

    acc_drug_impu_bce_loss = 0

    idx = 0

    train_data = TrainDataloader.dataset.imputation_data

    train_bar = tqdm(train_data, total=math.ceil(train_data.shape[0]/args.batch_size), desc=f'Epoch {epoch}', position=0)
    train_log = tqdm(total=0, bar_format='{desc}', position=1)
    shuffle_idx = np.random.permutation(train_data.shape[0])

    num_pos = torch.bincount(train_data.indices()[1])
    num_neg = torch.ones(train_data.shape[1], device=device)*train_data.shape[0] - num_pos
    num_pos, num_neg = num_pos[args.vocab_cum[1]:], num_neg[args.vocab_cum[1]:]

    weights = (torch.vstack([num_pos,  num_neg]).to(device) / (num_pos*num_neg*2+1e-6)).detach()

    Vd = args.vocab_size[1]
    loss_func_bce = WeightedFocalLoss(weights=weights, gamma=0)
    if 'wbce' in args.loss: 
        loss_func = WeightedFocalLoss(weights=weights, gamma=0)
    if 'wfl' in args.loss:
        loss_func = WeightedFocalLoss(weights=weights, gamma=5)
    if 'wkl' in args.loss:
        idf_weight = torch.log(train_data.shape[0] / (num_pos+1e-6))
        if not 'wbsam' in args.loss:
            loss_func = WeightedKLDivLoss(weights= idf_weight, gamma=args.gamma, weighted_by_sample=False)
        else:
            loss_func = WeightedKLDivLoss(weights= idf_weight, gamma=args.gamma, weighted_by_sample=True)
    elif 'kl' in args.loss:
        loss_func = nn.KLDivLoss(reduction='batchmean')
    
    if 'pnll' in args.loss:
        loss_func = nn.PoissonNLLLoss(log_input=False)

    for i in range(0, train_data.shape[0], args.batch_size):
        data_batch = []
        for j in range(0, min(args.batch_size, train_data.shape[0]-i)):
            data_batch.append(train_data[shuffle_idx[i+j]])
        data_batch = torch.vstack(data_batch).to_dense()

        optimizer.zero_grad()
        model.zero_grad()

        normalized_data_batch = normalize(data_batch, args.vocab_cum, args.bow_norm)

        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        if 'recon' in args.loss:
            total_loss.backward()

        optimizer.zero_grad()
        model.zero_grad()
        beta = model.get_beta()
        if 'bool' in args.loss:
            bows_drug = data_batch[:,args.vocab_cum[1]:].bool().float()
        else:
            bows_drug = data_batch[:,args.vocab_cum[1]:].float()
        normalized_data_batch[:,args.vocab_cum[1]:] = 0
        theta, _, = model.get_theta(normalized_data_batch)
        preds_drug = torch.mm(theta, beta['atc'])

        normalized_bows_drug = bows_drug / bows_drug.sum(-1).unsqueeze(-1) 
        if 'kl' in args.loss:
            if not 'wbsam' in args.loss:
                drug_impu_loss = loss_func(torch.log(preds_drug), normalized_bows_drug)
            else:
                drug_impu_loss = loss_func(torch.log(preds_drug), (normalized_bows_drug, bows_drug))
        else:
            drug_impu_loss = loss_func(preds_drug, bows_drug.bool().float())
        if 'pnll' in args.loss:
            drug_impu_loss = loss_func(preds_drug * bows_drug.sum(-1).unsqueeze(-1) , bows_drug)
        loss_bce = loss_func_bce(preds_drug.detach(), bows_drug.bool().float().detach())
 
        lamb = 50
        total_loss = drug_impu_loss * lamb 
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += recon_loss.item()
        acc_kl_theta_loss += kld_theta.item()
        acc_drug_impu_loss += drug_impu_loss.item()

        acc_drug_impu_bce_loss += loss_bce.item()

        idx += 1

        lr = optimizer.param_groups[0]['lr']
        cur_loss = round(acc_loss / idx, 2)
        cur_kl_theta = round(acc_kl_theta_loss / idx, 2)
        cur_drug_impu_loss = round(acc_drug_impu_loss / idx, 6)

        cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        if idx % args.log_interval == 0 and idx > 0:
            train_bar.update(args.log_interval)
            train_log.set_description_str(f'DI loss: {cur_drug_impu_loss } . (BCE:{round(acc_drug_impu_bce_loss /idx, 6)}) KL_theta: {cur_kl_theta} . Rec_loss: {cur_loss} . NELBO: {cur_real_loss}')

    train_bar.close()
    train_log.close()

    cur_loss = round(acc_loss / idx, 2)
    cur_kl_theta = round(acc_kl_theta_loss / idx, 2)

    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('Epoch----->{} .time: {:.2f} s. LR: {} .. KL_theta: {} .. Rec_loss: {} ... NELBO: {}'.
          format(epoch, time.time()-epoch_start_time, optimizer.param_groups[0]['lr'], cur_kl_theta,
                               cur_loss, cur_real_loss))
    print('*' * 80)



def evaluate(m, tq=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        beta = m.get_beta()

        nll_loss = np.zeros(len(args.code_types))
        acc_loss = 0
        cnt = 0
        for idx, (sample_batch, index) in tqdm(enumerate(TestDataloader)):
            # Data is already dense and batched from collate_fn
            data_batch_1 = sample_batch['Data_1'].float().to(device)
            if data_batch_1.dim() > 2:
                data_batch_1 = data_batch_1.squeeze(1)
            normalized_data_batch = normalize(data_batch_1, args.vocab_cum, args.bow_norm)

            theta, _, = m.get_theta(normalized_data_batch)

            data_batch_2 = sample_batch['Data_2'].float().to(device)
            if data_batch_2.dim() > 2:
                data_batch_2 = data_batch_2.squeeze(1)

            nll = m.decode(theta, beta, data_batch_2)
            recon_loss = nll.sum()

            loss = recon_loss
            acc_loss += loss.item()
            nll_loss += nll.data.cpu().numpy()
            cnt += 1
        cur_loss = round(acc_loss / cnt, 2)
        cur_nll = np.around(nll_loss / cnt, 2)
        print('*' * 100)
        print(f'Test Negative Log Likelihood: {list(cur_nll)}, sum: {cur_loss}')
        if args.drug_imputation:
            imputation_data = TestDataloader.dataset.imputation_data
            imp_prec, imp_recall, imp_f1, imp_acc = 0, 0, 0, 0
            imp_cnt = 0
            hits_at_k = torch.zeros(args.vocab_size[1], device=device)
            for i in tqdm(range(0, imputation_data.shape[0], args.batch_size)):
                data_batch = []
                for j in range(min(args.batch_size, imputation_data.shape[0]-i)):
                    data_batch.append(imputation_data[i+j])
                data_batch = torch.vstack(data_batch).to_dense()
                normalized_data_batch = normalize(data_batch, args.vocab_cum, args.bow_norm)

                bows_drug = data_batch[:,args.vocab_cum[1]:].bool().float()
                normalized_data_batch[:,args.vocab_cum[1]:] = 0
                theta, _, = m.get_theta(normalized_data_batch)
                preds_drug = torch.mm(theta, beta['atc'])
                k_idx = torch.argsort(preds_drug,dim=1)[:,-args.impute_k:]

                hits_at_k_i = (torch.zeros_like(bows_drug).scatter_(1,k_idx,1) * bows_drug).sum(0)
                hits_at_k += hits_at_k_i

                patient_impute_k = 5
                hits_at_k_pa = torch.gather(bows_drug,1,k_idx[:,-patient_impute_k:]).sum(1)
                num_pos = bows_drug.sum(1)
                imp_prec_i = (hits_at_k_pa / patient_impute_k)
                imp_recall_i = (hits_at_k_pa / num_pos)
                imp_f1_i = (2* imp_prec_i * imp_recall_i / (imp_prec_i + imp_recall_i+1e-7))

                imp_prec += imp_prec_i.sum().item()
                imp_recall += imp_recall_i.sum().item()
                imp_f1 += imp_f1_i.sum().item()
                imp_acc += (hits_at_k_pa > 0).sum().item()
                imp_cnt += data_batch.shape[0]

            test_total  = TestDataset.imputation_drug_freq
            train_total = TrainDataset.imputation_drug_freq
            test_total *= train_total.bool()
            hits_at_k *= train_total.bool()
            code_order = torch.argsort(test_total)
            hits_at_k_sorted = hits_at_k[code_order]
            test_total_sorted = test_total[code_order]

            acc_at_k = hits_at_k / (test_total+1e-7)
            acc_at_k_sorted = acc_at_k[code_order]
            acc_at_k_mean = hits_at_k.sum() / test_total.sum()
            cur_loss = - acc_at_k_mean

            st = torch.nonzero(test_total[code_order])[0,0].item()
            print((st, args.vocab_size[1]))
            l = int(np.ceil((args.vocab_size[1] - st) / 5))
            acc_at_k_partition = []
            for i in range(st, args.vocab_size[1], l):
                acc_at_k_partition.append(round((hits_at_k_sorted[i:i+l].sum() / test_total_sorted[i:i+l].sum()).item(),6))

            acc_at_k_partition_10 = []
            l = int(np.ceil((args.vocab_size[1] - st) / 10))
            for i in range(st, args.vocab_size[1], l):
                acc_at_k_partition_10.append(round((hits_at_k_sorted[i:i+l].sum() / test_total_sorted[i:i+l].sum()).item(),6))

            print(f'Drug Imputation: accuracy@k: {acc_at_k_partition}')
            print(f'  overall accuracy@k: {hits_at_k.sum() / test_total.sum()}')
            print(f'Drug Imputation: accuracy@k: {acc_at_k_partition_10}')
            print(f'patient-wise: precicsion@k: {imp_prec/imp_cnt}, recall@k: {imp_recall/imp_cnt}, F1-score@k: {imp_f1/imp_cnt}')
        print('*' * 100)

        TQ = TC = TD = 0

        if tq:
            train_data = TrainDataset.stack_bow

            TC = {}
            for i, c in enumerate(args.code_types):
                beta_i = beta[c].data.cpu().numpy()
                train_data_i = train_data[c]
                TC[c] = get_topic_coherence(beta_i, train_data_i, 3)
                TC[c] = round(TC[c], 3)

            print(f'Topic Coherence (top 3 / type) is: {TC}, mean: {np.mean(list(TC.values()))}')

            TD = {}
            for i, c in enumerate(args.code_types):
                TD[c] = get_topic_diversity(beta[c].data.cpu().numpy(), 3)
                TD[c] = round(TD[c], 3)
            print(f'Topic Diversity (top 3 / type) is: {TD}, mean: {np.mean(list(TD.values()))}')

            TQ = np.array(list(TD.values())) * np.array(list(TC.values()))
            print(f'Topic Quality is: {TQ}, mean: {np.mean(TQ)}')
            print('#' * 100)

        return cur_loss, TQ, TC, TD

if args.mode == 'train':
    ## train model on data
    best_epoch = 0
    best_val_metric = 1e9
    all_val_metrics = []

    for epoch in range(1, args.epochs+1):
        if args.drug_imputation: 
            train_DI(epoch)
        else:
            train(epoch)
        val_metric, tq, tc, td = evaluate(model, args.tq)
        if val_metric < best_val_metric or not os.path.exists(ckpt):
            with open(ckpt, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_epoch = epoch
            best_val_metric = val_metric
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (
                    len(all_val_metrics) > args.nonmono and val_metric > min(all_val_metrics[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        all_val_metrics.append(val_metric)
    with open(ckpt, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model = model.to(device)
    val_ppl, tq, tc, td = evaluate(model, args.tq)
    print('Best Epoch: ' + str(best_epoch))
else:
    model.eval()
    val_ppl, tq, tc, td = evaluate(model, args.tq)


model.eval()

print('save embeddings...')
beta = model.get_beta()
for i, code in enumerate(args.code_types):
    beta[code] = beta[code].detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "beta_"+code+".npy"), beta[code])
    rho_i = model.rho[code].detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "rho_"+code+".npy"), rho_i)

if args.sharea:
    alpha = model.alphas.weight.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "alpha.npy"), alpha)
else:
    for i, code in enumerate(args.code_types):
        alpha_i = model.alphas[code].weight.detach().cpu().numpy()
        np.save(os.path.join(args.save_path, "alpha_"+code+".npy"), alpha_i)


#####
print('save theta of training set...')

index_list = []
for idx, (sample_batch, index) in tqdm(enumerate(TrainDataloader)):
    index_list.append(index.cpu().numpy())
    # Data is already dense and batched from collate_fn
    data_batch = sample_batch["Data"].float().to(device)
    if data_batch.dim() > 2:
        data_batch = data_batch.squeeze(1)
    normalized_data_batch = normalize(data_batch, args.vocab_cum, args.bow_norm)
    theta, _, = model.get_theta(normalized_data_batch)
    theta = theta.detach().cpu().numpy()
    saved_folder = os.path.join(args.save_path, "theta_train")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    np.save(saved_theta, theta)
saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)

print('save theta of test set...')

index_list = []
for idx, (sample_batch, index) in tqdm(enumerate(TestDataloader)):
    index_list.append(index.cpu().numpy())
    # Data is already dense and batched from collate_fn
    data_batch = sample_batch['Data'].float().to(device)
    if data_batch.dim() > 2:
        data_batch = data_batch.squeeze(1)
    normalized_data_batch = normalize(data_batch, args.vocab_cum, args.bow_norm)

    theta, _, = model.get_theta(normalized_data_batch)
    theta = theta.detach().cpu().numpy()
    saved_folder = os.path.join(args.save_path, "theta_test")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    np.save(saved_theta, theta)
saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)

