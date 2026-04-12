"""
Training Script for GAT-ETM on Vietnamese EHR Data

This script is adapted from main_getm.py for Vietnamese data.
Main changes:
1. Custom collate_fn for sparse tensors (PyTorch 2.x compatibility)
2. Paths configured for data_vn/ and embed_vn/
3. Simplified argument handling

Usage:
    # Basic training
    python train_vn.py --epochs 50 --num_topics 50

    # With topic quality evaluation
    python train_vn.py --epochs 50 --num_topics 50 --tq

    # Drug imputation mode
    python train_vn.py --epochs 50 --drug_imputation --loss wkl --gamma 1.0

    # Evaluation only
    python train_vn.py --mode eval --load_from results_vn/model.pt
"""

import argparse
import torch
import numpy as np
import os
import time
import math
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim

from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity
from dataset import NontemporalDataset, normalize


def custom_collate(batch):
    """
    Custom collate function for sparse tensors.
    Required for PyTorch 2.x compatibility.
    """
    samples = [item[0] for item in batch]
    indices = [item[1] for item in batch]

    # Convert sparse tensors to dense and stack
    data_list = [s['Data'].to_dense() for s in samples]
    data_batch = torch.stack(data_list)

    result = {'Data': data_batch}

    # Handle test phase data
    if 'Data_1' in samples[0]:
        data_1_list = [s['Data_1'].to_dense() for s in samples]
        data_2_list = [s['Data_2'].to_dense() for s in samples]
        result['Data_1'] = torch.stack(data_1_list)
        result['Data_2'] = torch.stack(data_2_list)

    return result, torch.tensor(indices)


def parse_args():
    parser = argparse.ArgumentParser(description='GAT-ETM Training on Vietnamese EHR Data')

    # Data paths (configured for VN data)
    parser.add_argument('--data_path', type=str, default='data_vn/',
                        help='Directory containing data')
    parser.add_argument('--graph_path', type=str,
                        default='embed/augmented_icdatc_graph_256_renumbered_by_vocab.pkl',
                        help='Path to knowledge graph')
    parser.add_argument('--embed_path', type=str,
                        default='embed/augmented_icdatc_embed_8_20_10_256_by_vocab.pkl',
                        help='Path to graph embeddings')
    parser.add_argument('--save_path', type=str, default='results_vn/',
                        help='Path to save results')

    # Model architecture
    parser.add_argument('--num_topics', type=int, default=50,
                        help='Number of topics')
    parser.add_argument('--rho_size', type=int, default=256,
                        help='Dimension of code embeddings')
    parser.add_argument('--emb_size', type=int, default=256,
                        help='Dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=128,
                        help='Hidden size of theta encoder')
    parser.add_argument('--theta_act', type=str, default='relu',
                        help='Activation function (relu, tanh, softplus)')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of layers for theta encoder')
    parser.add_argument('--enc_drop', type=float, default=0.1,
                        help='Dropout rate on encoder')
    parser.add_argument('--upper', type=int, default=10,
                        help='Upper boundary for Gaussian variance')
    parser.add_argument('--lower', type=int, default=-10,
                        help='Lower boundary for Gaussian variance')
    parser.add_argument('--sharea', action='store_true',
                        help='Share alpha across code types')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--lr_factor', type=float, default=2.0,
                        help='LR reduction factor')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='Weight decay')
    parser.add_argument('--clip', type=float, default=2.0,
                        help='Gradient clipping')
    parser.add_argument('--anneal_lr', type=int, default=1,
                        help='Whether to anneal LR')
    parser.add_argument('--nonmono', type=int, default=10,
                        help='Non-monotonic patience')
    parser.add_argument('--bow_norm', type=bool, default=True,
                        help='Normalize BoW')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Drug imputation
    parser.add_argument('--drug_imputation', action='store_true',
                        help='Enable drug imputation mode')
    parser.add_argument('--dc_thr', type=int, default=5,
                        help='Drug count threshold')
    parser.add_argument('--impute_k', type=int, default=5,
                        help='Top-k for imputation')
    parser.add_argument('--loss', type=str, default='wkl',
                        help='Loss function (wkl, wbce, wfl, kl)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma for weighted loss')

    # Evaluation
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    parser.add_argument('--load_from', type=str, default='',
                        help='Checkpoint to load')
    parser.add_argument('--tq', action='store_true',
                        help='Compute topic quality metrics')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='Evaluation batch size')

    # Hardware
    parser.add_argument('--gpu_device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')

    # Logging
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Log interval')

    return parser.parse_args()


def load_data(args, device):
    """Load all required data."""
    print("Loading data...")

    # Load metadata
    metadata = np.loadtxt(os.path.join(args.data_path, 'metadata.txt'), dtype=str)
    code_types = metadata[0]
    vocab_size = [int(v) for v in metadata[1]]
    train_embeddings = [int(v) for v in metadata[2]]
    embedding_paths = metadata[3]

    args.code_types = code_types
    args.vocab_size = vocab_size
    args.vocab_cum = np.cumsum([0] + vocab_size)
    args.train_embeddings = train_embeddings

    print(f"  Code types: {code_types}")
    print(f"  Vocab sizes: {vocab_size}")

    # Load graph
    print(f"  Loading graph from {args.graph_path}...")
    args.graph = pickle.load(open(args.graph_path, 'rb'))

    # Load graph embeddings
    print(f"  Loading embeddings from {args.embed_path}...")
    args.graph_embed = pickle.load(open(args.embed_path, 'rb'))

    # Load individual embeddings
    embeddings = {}
    for i, c in enumerate(code_types):
        if embedding_paths[i] == '*':
            embeddings[c] = None
        else:
            emb = torch.from_numpy(np.load(embedding_paths[i])).to(device)
            embeddings[c] = emb
            print(f"  Loaded {c} embeddings: {emb.shape}")

    return embeddings


def create_dataloaders(args, device):
    """Create train and test dataloaders."""
    code_type_info = (args.code_types, args.vocab_size, args.vocab_cum)

    # Training dataset
    train_file = os.path.join(args.data_path, "bow_train.npy")
    train_dataset = NontemporalDataset(
        'train', train_file, code_type_info, device=device,
        drug_imputation=args.drug_imputation, drug_count_thr=args.dc_thr
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=custom_collate
    )

    # Test dataset
    test_file = os.path.join(args.data_path, "bow_test.npy")
    test_file_1 = os.path.join(args.data_path, "bow_test_1.npy")
    test_file_2 = os.path.join(args.data_path, "bow_test_2.npy")
    test_dataset = NontemporalDataset(
        'test', (test_file, test_file_1, test_file_2), code_type_info, device=device,
        drug_imputation=args.drug_imputation, drug_count_thr=args.dc_thr
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset, test_dataset


def train_epoch(model, train_loader, optimizer, args, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_kl = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (sample_batch, indices) in enumerate(pbar):
        optimizer.zero_grad()

        data_batch = sample_batch['Data'].to(device).float().squeeze(1)
        normalized_data = normalize(data_batch, args.vocab_cum, args.bow_norm)

        recon_loss, kld_theta = model(data_batch, normalized_data)
        loss = recon_loss + kld_theta

        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += recon_loss.item()
        total_kl += kld_theta.item()
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/n_batches:.2f}',
                'kl': f'{total_kl/n_batches:.2f}'
            })

    return total_loss / n_batches, total_kl / n_batches


def evaluate(model, test_loader, train_dataset, args, device, compute_tq=False):
    """Evaluate model."""
    model.eval()

    with torch.no_grad():
        beta = model.get_beta()

        total_nll = np.zeros(len(args.code_types))
        n_batches = 0

        for sample_batch, indices in tqdm(test_loader, desc='Evaluating'):
            # Document completion: use first half to predict second half
            data_1 = sample_batch['Data_1'].to(device).float().squeeze(1)
            data_2 = sample_batch['Data_2'].to(device).float().squeeze(1)

            normalized_data = normalize(data_1, args.vocab_cum, args.bow_norm)
            theta, _ = model.get_theta(normalized_data)

            nll = model.decode(theta, beta, data_2)
            total_nll += nll.data.cpu().numpy()
            n_batches += 1

        avg_nll = total_nll / n_batches
        print(f"Test NLL: {list(np.around(avg_nll, 2))}, sum: {np.sum(avg_nll):.2f}")

        # Topic quality metrics
        TC, TD, TQ = {}, {}, {}
        if compute_tq:
            train_data = train_dataset.stack_bow

            for i, c in enumerate(args.code_types):
                beta_c = beta[c].data.cpu().numpy()
                train_data_c = train_data[c]

                TC[c] = round(get_topic_coherence(beta_c, train_data_c, 3), 3)
                TD[c] = round(get_topic_diversity(beta_c, 3), 3)

            TQ = {c: TC[c] * TD[c] for c in args.code_types}

            print(f"Topic Coherence (top-3): {TC}, mean: {np.mean(list(TC.values())):.3f}")
            print(f"Topic Diversity (top-3): {TD}, mean: {np.mean(list(TD.values())):.3f}")
            print(f"Topic Quality: {TQ}, mean: {np.mean(list(TQ.values())):.3f}")

        return np.sum(avg_nll), TC, TD, TQ


def save_results(model, args, train_loader, test_loader):
    """Save model outputs."""
    print("Saving results...")

    model.eval()
    with torch.no_grad():
        beta = model.get_beta()

        # Save beta (topic-word distributions)
        for c in args.code_types:
            beta_np = beta[c].detach().cpu().numpy()
            np.save(os.path.join(args.save_path, f"beta_{c}.npy"), beta_np)
            print(f"  Saved beta_{c}.npy: {beta_np.shape}")

        # Save rho (code embeddings)
        for c in args.code_types:
            rho_np = model.rho[c].detach().cpu().numpy()
            np.save(os.path.join(args.save_path, f"rho_{c}.npy"), rho_np)
            print(f"  Saved rho_{c}.npy: {rho_np.shape}")

        # Save alpha (topic embeddings)
        if args.sharea:
            alpha_np = model.alphas.weight.detach().cpu().numpy()
            np.save(os.path.join(args.save_path, "alpha.npy"), alpha_np)
        else:
            for c in args.code_types:
                alpha_np = model.alphas[c].weight.detach().cpu().numpy()
                np.save(os.path.join(args.save_path, f"alpha_{c}.npy"), alpha_np)

    print("  Results saved!")


def main():
    args = parse_args()

    # Setup
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    # Load data
    embeddings = load_data(args, device)

    # Create dataloaders
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(args, device)

    # Create model
    print("\nCreating model...")
    if args.drug_imputation:
        from graph_etm_DI import GETM
    else:
        from graph_etm import GETM

    model = GETM(
        device=device,
        num_topics=args.num_topics,
        code_types=args.code_types,
        vocab_size=args.vocab_size,
        t_hidden_size=args.t_hidden_size,
        rho_size=args.rho_size,
        emsize=args.emb_size,
        theta_act=args.theta_act,
        graph=args.graph,
        graph_embed=args.graph_embed,
        embeddings=embeddings,
        train_embeddings=args.train_embeddings,
        enc_drop=args.enc_drop,
        upper=args.upper,
        lower=args.lower,
        share_alpha=args.sharea
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Checkpoint path
    ckpt_path = os.path.join(args.save_path, f"model_K{args.num_topics}.pt")

    if args.mode == 'eval':
        print(f"\nLoading model from {args.load_from}...")
        model.load_state_dict(torch.load(args.load_from, map_location=device))
        evaluate(model, test_loader, train_dataset, args, device, compute_tq=args.tq)
        return

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    best_epoch = 0
    all_losses = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_kl = train_epoch(model, train_loader, optimizer, args, device, epoch)

        print(f"Epoch {epoch} | Time: {time.time()-start_time:.1f}s | "
              f"Train Loss: {train_loss:.2f} | KL: {train_kl:.2f}")

        # Evaluate
        val_loss, tc, td, tq = evaluate(model, test_loader, train_dataset, args, device, compute_tq=args.tq)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model at epoch {epoch}")
        else:
            # Learning rate annealing
            if args.anneal_lr and len(all_losses) > args.nonmono:
                if val_loss > min(all_losses[:-args.nonmono]):
                    lr = optimizer.param_groups[0]['lr']
                    if lr > 1e-5:
                        optimizer.param_groups[0]['lr'] /= args.lr_factor
                        print(f"  LR reduced to {optimizer.param_groups[0]['lr']:.6f}")

        all_losses.append(val_loss)
        print()

    # Load best model and final evaluation
    print(f"\nBest epoch: {best_epoch}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    evaluate(model, test_loader, train_dataset, args, device, compute_tq=args.tq)

    # Save results
    save_results(model, args, train_loader, test_loader)

    print("\nTraining complete!")
    print(f"Model saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
