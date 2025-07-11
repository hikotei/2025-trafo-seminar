# Standard library imports
import os
import gc
import sys
import time
import math
import pickle
import random
import string
import argparse
import datetime

# Third-party imports
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

# Check for Metal availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal backend")
else:
    device = torch.device("cpu")
    print("Metal not available, using CPU")


class simu_xform:
    """
    This class is a simulation of the transformer self-attention dynamics.
    xform is short for transformer.

    contains the following class methods:
        - __init__: initialize the simulation
        - generate_KQV: generate the key-query-value matrices
        - update_IP: update the inner product matrix
        - check_stats: check the statistics of the simulation
        - compute_hist1: compute the histogram of the inner product matrix
        - compute_hist_ensemble: compute the histogram of the inner product matrix for a specific particle pair
        - compute_hist_all: compute the histogram of the inner product matrix for all particle pairs
        - count_clusters: count the number of clusters in the simulation
        - step_1: execute one time step of the simulation

    later on objects of this class will be used
    to run the simulation for a range of beta values.
    using the methods :
        - do_single_beta: run the simulation for a single beta value
        - do_results: run the simulation for a range of beta values
        - plot_pickle: plot the results of the simulation from a pickle file
    """

    def __init__(self, args, beta, V=None, BF=None):
        """
        Initialize the transformer self-attention particle system simulation.

        Models tokens as particles on a high-dimensional sphere that interact through
        self-attention dynamics. Each particle represents a token embedding, and the
        system evolves according to attention-weighted updates.

        Args:
            args: Configuration object containing simulation parameters
                - batch: Number of independent systems to simulate in parallel
                - ntokens: Number of tokens/particles per system
                - dmodel: Dimensionality of token embeddings (sphere dimension)
                - step: Time step size for dynamics
                - use_softmax: Whether to use softmax normalization in attention
            beta: Temperature parameter controlling attention sharpness
                - Higher beta = sharper attention, more clustering
                - Lower beta = diffuse attention, less clustering
            V: Optional pre-computed value matrix (for reusing across simulations)
            BF: Optional pre-computed bilinear form matrix (key-query interaction)

        The system tracks particle positions (M), inner products (IP), softmax weights (SM),
        and evolves particles according to self-attention dynamics on the sphere.
        """
        batch, ntokens, dmodel = args.batch, args.ntokens, args.dmodel
        self.args = args
        self.beta = beta
        # Normalize step so that we don't take more than 0.1 radian moves
        if (not args.use_softmax) and (not args.rawstep):
            max_val = max(np.sqrt(beta) * np.exp(beta - 1 / 2), 1)
            eta = args.step / max_val

            # cursor suggestion :
            # max_val = torch.max(torch.sqrt(beta) * torch.exp(beta - 1 / 2), torch.tensor(1.0))
            # eta = args.step / max_val.item()

            print(f"Corrected step = {eta}")
        else:
            eta = args.step

        self.eta = eta
        # Matrix of all particles
        self.M = torch.zeros(batch, ntokens, dmodel, requires_grad=False).to(device)
        # Holder for inner products
        self.IP = torch.zeros(batch, ntokens, ntokens, requires_grad=False).to(device)
        # Holder for softmax
        self.SM = torch.zeros(batch, ntokens, ntokens, requires_grad=False).to(device)
        # Holder for output
        self.M2 = torch.randn(batch, ntokens, dmodel, requires_grad=False).to(device)
        self.M3 = torch.empty(batch, ntokens, dmodel, requires_grad=False).to(device)
        # Random Bilinear form (Q^T K thing)
        self.BF = BF
        self.V = V
        # Generate matrices
        self.generate_KQV()
        # Initialize particles
        torch.nn.functional.normalize(self.M2, dim=2, out=self.M)

    def generate_KQV(self):
        """
        Generate key-query-value matrices that define the attention mechanism.

        Creates the bilinear form (BF) for key-query interactions and value matrix (V)
        that determine how particles influence each other. Different modes create:
        - Identity matrices (standard case)
        - Random Gaussian matrices (introduces randomness)
        - Gaussian Orthogonal Ensemble (GOE) matrices (structured randomness)
        - Wigner matrices (symmetric random matrices)

        The BF matrix determines the attention pattern geometry, while V matrix
        controls how attended information is combined. These matrices can be:
        - Regenerated each step (simulating different layers)
        - Fixed per beta (temperature-dependent but layer-independent)
        - Fixed globally (same across all conditions)
        """
        args = self.args
        batch, ntokens, dmodel = args.batch, args.ntokens, args.dmodel
        # In noanneal=2 mode we do not touch BF
        if (args.noanneal == 2) and (self.BF != None) and (self.V != None):
            print("[DEBUG] noanneal=2, reusing BF with id = ", id(self.BF))
            print("[DEBUG] noanneal=2, reusing V with id = ", id(self.V))
            return

        if args.randomKQ == 1:
            # with pp = 1 the matrix mult should broadcast as if BF is it's the same BF for all systems
            pp = 1 if (args.noanneal > 0) else batch
            BF1 = torch.randn(pp, dmodel, dmodel, requires_grad=False).to(device)
            self.BF = BF1.matmul(
                torch.randn(pp, dmodel, dmodel, requires_grad=False).to(device)
            )
            self.BF /= np.sqrt(dmodel)
            normV = np.sqrt(dmodel)

        elif args.randomKQ == 2:
            # This was a wrong way to do Wigner (forgot about diagonal being different)
            pp = 1 if (args.noanneal > 0) else batch
            BF1 = torch.randn(pp, dmodel, dmodel, requires_grad=False).to(device)
            norma = torch.eye(dmodel) * (2 - math.sqrt(2)) + torch.ones(dmodel).to(
                device
            ) * math.sqrt(2)
            self.BF = (BF1 + BF1.transpose(1, 2)) / norma
            normV = np.sqrt(dmodel)

        elif args.randomKQ == 3:
            # This is proper GOE matrix
            pp = 1 if (args.noanneal > 0) else batch
            BF1 = torch.randn(pp, dmodel, dmodel, requires_grad=False).to(device)
            norma = math.sqrt(2)
            self.BF = (BF1 + BF1.transpose(1, 2)) / norma
            if False:
                L, _ = torch.linalg.eigh(self.BF[0, :, :])
                print("[debug] GOE BF eigenvalues of 0th system = ", L)
            normV = np.sqrt(dmodel + 1)

        else:
            # This creates [1, dmodel, dmodel] tensor. Hopefully batch dimension being of length {1}
            # leads to correct broadcasting...
            self.BF = torch.unsqueeze(
                torch.eye(dmodel, dmodel, requires_grad=False, device=device), 0
            )
            normV = 1

        # randomV = 1 means random V. 2 means gradient ascent (V=+cB), 3 descent (V=-cB)

        if self.args.randomV == 1:
            pp = 1 if (args.noanneal > 0) else batch
            self.V = torch.randn(pp, dmodel, dmodel, requires_grad=False).to(device)
            self.V = torch.matmul(self.V, self.V.transpose(-2, -1))
            self.V /= np.sqrt(dmodel)

        elif self.args.randomV == 2:
            self.V = self.BF / normV

        elif self.args.randomV == 3:
            self.V = -self.BF / normV

        else:
            # identity case
            # This creates [1, dmodel, dmodel] tensor. Hopefully batch dimension being of length {1}
            # leads to correct broadcasting...
            self.V = torch.unsqueeze(
                torch.eye(dmodel, dmodel, requires_grad=False, device=device), 0
            )

    # Compute inner products from particle positions M
    def update_IP(self):
        """
        Update the inner product matrix between all pairs of particles.

        Computes the dot products between all token embeddings, which determines
        the similarity structure of the system. Inner products close to 1 indicate
        particles that are nearly aligned (clustered), while values near 0 indicate
        orthogonal particles. This matrix is central to measuring clustering behavior.

        Updates self.IP with shape (batch, ntokens, ntokens) where IP[i,j,k] is the
        inner product between particles j and k in system i.
        """
        torch.matmul(self.M, self.M.transpose(1, 2), out=self.IP)

    # Compute density at zero and average number of clusters.
    def check_stats(self):
        """
        Analyze the clustering statistics of the particle system.

        Computes key metrics that quantify how much clustering has occurred:
        - For plotdim=True: Counts the effective dimensionality by looking at
          eigenvalues of the inner product matrix (lower dim = more clustering)
        - For plotdim=False: Counts fraction of particle pairs with inner product > 0.999
          (high fraction = many particles are nearly aligned/clustered)

        Returns:
            dens: Density measure - either effective dimension or clustering fraction
            median_clust: Median number of clusters across all systems (if computed)

        This provides the main observable for studying phase transitions in attention
        dynamics as temperature (beta) varies.
        """
        self.update_IP()
        batch = self.args.batch
        if self.args.plotdim:
            try:
                # eigenvals of a complex Hermitian or real symmetric matrix
                eigv = torch.linalg.eigvalsh(self.IP)  # batch x ntoken
            except Exception as inst:
                print("[ERROR] linalg.eigvalsh failed: ", inst)
                dens = 0
            else:
                # print('[debug] eigv = ', eigv);
                dens = (eigv > 0.001).flatten().sum() / batch
        else:
            # Earlier versions had a bug where we also counted i=j particles into the sum.
            mask = ~torch.eye(self.args.ntokens).bool().expand(batch, -1, -1)
            inner_prods = self.IP[mask]  # this should be automatically flattened
            dens = (inner_prods > 0.999).sum() / len(inner_prods)

        ### This is a version that counts i=j particles as well. Use it to check for numerical instabilities
        # inner_prods = self.IP.flatten();
        # dens = (inner_prods > 0.999).sum() / len(inner_prods);

        if self.args.cluster_sizes:
            time0 = time.time()
            list_clust = []
            for i in range(batch):
                T = self.IP[i, :, :]
                T1 = (torch.real(T) > 0.9999).type(torch.FloatTensor)
                nr_clust = torch.linalg.matrix_rank(T1, hermitian=False)
                # T1L, T1V = torch.linalg.eig(T1);
                # cluster_sizes = (T1L[torch.abs(T1L)>1e-3])
                list_clust.append(nr_clust)

            median_clust = np.median(np.asarray(list_clust))
            time1 = time.time()
            print(f"[debug] check_stats() took {(time1 - time0):.2f} sec")
        else:
            median_clust = 0

        return dens, median_clust

    def compute_hist1(self, idx=0):
        """
        Compute detailed clustering statistics for a single system.

        Args:
            idx: Index of the system to analyze (default: 0)

        Returns:
            dens: Histogram density of inner products
            bins: Histogram bin edges
            cluster_sizes: Sizes of identified clusters

        Provides fine-grained analysis of the clustering structure in one system,
        including the distribution of inner products and cluster size information.
        Useful for detailed inspection of clustering patterns.
        """
        dens, bins = torch.histogram(
            self.IP[idx, :, :].flatten().cpu(), 250, density=True
        )
        T = self.IP[idx, :, :]
        T1 = (torch.real(T) > 0.9999).type(torch.FloatTensor)
        nr_clust = torch.linalg.matrix_rank(T1, hermitian=False)
        T1L, T1V = torch.linalg.eig(T1)
        T1L = torch.abs(T1L)
        cluster_sizes = T1L[T1L > 0.001]
        return dens, bins, cluster_sizes

    # Return histogram of inner products of (X_1,X_2) across different systems
    def compute_hist_ensemble(self):
        """
        Compute histogram of inner products for specific particle pairs across ensemble.

        Focuses on correlations between consecutive particle pairs (X_1,X_2), (X_3,X_4), etc.
        across all systems in the ensemble. This provides insight into typical pairwise
        clustering behavior without including all possible pairs.

        Returns:
            dens: Histogram density of selected inner products
            bins: Histogram bin edges

        Useful for studying ensemble-averaged clustering statistics while focusing on
        natural particle pairings.
        """
        ntokens = self.args.ntokens
        pairs = [self.IP[:, x, x + 1].flatten() for x in range(0, ntokens, 2)]
        all_corr = torch.cat(pairs).cpu()
        dens, bins = torch.histogram(all_corr, 250, density=True)
        return dens, bins

    # Finally, return giant histo of everything
    def compute_hist_all(self):
        """
        Compute histogram of all inner products across all systems.

        Returns:
            dens: Histogram density of all inner products
            bins: Histogram bin edges

        Provides the most comprehensive view of clustering by including every pairwise
        inner product across all systems. Shows the full distribution of particle
        similarities in the ensemble.
        """
        dens, bins = torch.histogram(self.IP.flatten().cpu(), 500, density=True)
        return dens, bins

    def count_clusters(self):
        """
        Count the number of distinct clusters in each system.

        Returns:
            Array of cluster counts for each system in the batch

        Determines how many separate groups of nearly-aligned particles exist in each
        system by analyzing the rank of the thresholded inner product matrix.
        Lower cluster counts indicate more particles have merged into fewer groups.
        """
        batch, ntokens = self.args.batch, self.args.ntokens
        list_clust = []
        for i in range(batch):
            T = self.IP[i, :, :]
            T1 = (torch.real(T) > 0.9999).type(torch.FloatTensor)
            nr_clust = torch.linalg.matrix_rank(T1, hermitian=False)
            # T1L, T1V = torch.linalg.eig(T1);
            # cluster_sizes = (T1L[torch.abs(T1L)>1e-3])
            list_clust.append(nr_clust)

        return np.asarray(list_clust)

    def step_1(self, time):
        """
        Execute one time step of the self-attention dynamics.

        Args:
            time: Current simulation time

        Returns:
            Updated time (time + step_size)

        Implements the core dynamics of transformer self-attention as a particle system:
        1. Apply bilinear form (key-query interaction) to particles
        2. Compute attention weights via softmax or normalized exponential
        3. Apply attention-weighted combination (value transformation)
        4. Update particle positions while maintaining normalization on sphere

        This is the heart of the simulation, where particles interact through attention
        and evolve according to the resulting forces. Higher beta leads to sharper
        attention and stronger clustering forces.
        """
        # If you want non-identity bilinear form keep at True
        if self.args.randomKQ > 0:
            torch.matmul(self.M, self.BF, out=self.M2)
        else:
            self.M2.copy_(self.M)  # store M into M2

        torch.matmul(self.M2, self.M.transpose(1, 2), out=self.SM)
        self.SM *= self.beta
        # Recompute IP for the case when BF != identity
        # Later: removed, use update_IP() for this
        # torch.matmul(self.M,self.M.transpose(1,2), out=self.IP);

        # Compute softmax
        if args.use_softmax:
            # Old, numerically unstable implementation
            # torch.exp(self.SM, out=self.SM);
            # summed = torch.sum(self.SM, dim=2, keepdim=True);
            # self.SM /= summed;

            # Another method (it uses 220% CPU and GPU util drops to 97%
            self.SM = torch.nn.functional.softmax(self.SM, dim=2)

        # Another method (matches the above in utilization)
        # maxx, _ = self.SM.max(dim=2, keepdim=True);
        # self.SM -= maxx;
        # torch.exp(self.SM, out=self.SM);
        # summed = torch.sum(self.SM, dim=2, keepdim=True);
        # self.SM /= summed;

        else:
            torch.exp(self.SM, out=self.SM)
            self.SM /= self.args.ntokens

        # Multiply softmax and particles
        torch.matmul(self.SM, self.M, out=self.M2)
        # Add to orig. particles
        # Set to False to enable random Value projection
        self.M2 *= self.eta
        if self.args.randomV > 0:
            # Apply random rotation (optional?)
            torch.matmul(self.M2, self.V, out=self.M3)
            self.M2.copy_(self.M3)

        self.M2 += self.M
        # Update the particle matrix
        torch.nn.functional.normalize(self.M2, dim=2, out=self.M)
        return time + self.eta


def do_single_beta(args, beta_idx, ret_dict, V=None, BF=None):
    """
    Run the complete simulation for a single temperature (beta) value.

    Args:
        args: Simulation configuration parameters
        beta_idx: Index of the beta value to simulate
        ret_dict: Dictionary containing simulation setup (beta values, time points, storage)
        V: Optional pre-computed value matrix
        BF: Optional pre-computed bilinear form matrix

    Returns:
        Updated ret_dict with simulation results

    Executes the full time evolution for one temperature, recording clustering statistics
    at specified time checkpoints. This captures the dynamics of how clustering evolves
    over time for a fixed attention sharpness (beta), which is crucial for understanding
    the phase diagram of the system.
    """
    beta = ret_dict["betas_list"][beta_idx]
    s = simu_xform(args, beta, V=V, BF=BF)
    cur_ckpt = 0
    cur_time = 0.0
    time_st = time.time()
    dt = ret_dict["density_tensor"]
    ct = ret_dict["cluster_tensor"]
    regen_time = args.regen_period if args.regen_period > 0.0 else float("inf")
    max_ckpt = len(ret_dict["times_list"])
    while True:
        cur_time = s.step_1(cur_time)
        if cur_ckpt < max_ckpt and cur_time >= ret_dict["times_list"][cur_ckpt]:
            time1 = time.time()
            dens, clust = s.check_stats()
            while cur_ckpt < max_ckpt and cur_time >= ret_dict["times_list"][cur_ckpt]:
                dt[beta_idx, cur_ckpt] = dens
                ct[beta_idx, cur_ckpt] = clust
                if cur_ckpt % 10 == 0:
                    print(
                        f" .... (beta = {beta:.2f}, time = {cur_time:.2f}). Time / ckpt = {(time1 - time_st) / 60:.2f} min"
                    )
                    print(
                        f" .... dens = {dens:.2g}, median nr clusters = {clust:.2g}",
                        flush=True,
                    )
                cur_ckpt += 1

            time_st = time1

        if cur_time >= args.maxtime or cur_ckpt >= max_ckpt:
            break

        if cur_time > regen_time:
            s.generate_KQV()
            regen_time += args.regen_period

    del s
    torch.cuda.empty_cache()
    gc.collect()
    return ret_dict


def do_results(args):
    """
    Execute the complete phase diagram simulation across all temperature values.

    Args:
        args: Complete simulation configuration containing all parameters

    Returns:
        Dictionary containing full simulation results including:
        - beta values (temperature range)
        - time points
        - density tensor (clustering measure vs beta and time)
        - cluster tensor (cluster count vs beta and time)

    This is the main simulation driver that maps out the phase diagram of transformer
    self-attention dynamics. It explores how clustering behavior changes as a function
    of both temperature (beta) and time, revealing phase transitions between different
    clustering regimes. The results show where and when tokens begin to cluster together
    in the attention mechanism.
    """
    # Change default tensor type to float32 for MPS compatibility
    torch.set_default_dtype(torch.float32)
    x = torch.zeros(3, 3)
    print("Checking default tensor type = ", x.dtype)
    del x
    ret_dict = {}
    betas_list = torch.linspace(args.betamin, args.betamax, args.betas)
    times_list = torch.linspace(0, args.maxtime, 200)
    ret_dict.update({"betas_list": betas_list})
    ret_dict.update({"times_list": times_list})
    density_tensor = torch.zeros(len(betas_list), len(times_list))
    ret_dict.update({"density_tensor": density_tensor})
    cluster_tensor = torch.zeros(len(betas_list), len(times_list))
    ret_dict.update({"cluster_tensor": cluster_tensor})
    time_st = time.time()
    Vsave = None
    BFsave = None
    if args.noanneal >= 2:
        s = simu_xform(args, 0)
        Vsave = s.V
        BFsave = s.BF
        del s
        torch.cuda.empty_cache()
        gc.collect()

    for i, beta in tqdm.tqdm(
        enumerate(betas_list),
        total=len(betas_list),
        disable=args.disable_tqdm,
        desc=f"Simu d={args.dmodel}, n={args.ntokens}, randomV={args.randomV}",
    ):
        print(
            f"[START] Step {i}. beta = {beta:.3f}. Start time = {str(datetime.datetime.now())}..."
        )
        time0 = time.time()
        ret_dict = do_single_beta(args, i, ret_dict, V=Vsave, BF=BFsave)
        time1 = time.time()
        est_time = float(time1 - time_st) / (i + 1) * (len(betas_list) - i - 1)
        if False:
            with open(fname_prefix + "_tmp.pkl", "wb") as f:
                pickle.dump(ret_dict, f)
        print(
            f"[END] Step {i}. beta = {beta:.3f}. End time = {str(datetime.datetime.now())}."
        )
        print(
            f"[END] Elapsed = {(time1 - time0) / 60:.2f} min. ETA = {est_time / 60:.2f} min"
        )

    return ret_dict


def plot_pickle(fname):
    """
    Create phase diagram plots from saved simulation results.

    Args:
        fname: Path to pickle file containing simulation results

    Generates publication-quality phase diagrams showing clustering behavior as a function
    of temperature (beta) and time. Creates both linear and logarithmic scale plots that
    reveal:
    - Phase boundaries between different clustering regimes
    - Critical temperatures where clustering begins
    - Time scales for cluster formation
    - Overall phase diagram structure

    The plots use color maps to show clustering density, with different colors indicating
    different levels of clustering. This visualizes the "phase transitions" in transformer
    attention dynamics.
    """
    with open(fname, "rb") as f:
        ret = pickle.load(f)

    args = ret["args"]
    betas_list = ret["betas_list"]
    times_list = ret["times_list"]
    density_tensor = ret["density_tensor"]
    cluster_tensor = ret["cluster_tensor"]
    # im=ax.imshow(np.log(density_tensor), aspect='auto')
    # fig.colorbar(im, ax=ax, label='log(density(0))')
    Y = betas_list.reshape(-1, 1).expand(-1, times_list.shape[0])
    X = times_list.reshape(1, -1).expand(betas_list.shape[0], -1)
    # for cscale in ['linear']:
    for cscale in ["log", "linear"]:
        fig, ax = plt.subplots()

        suffix_n = cscale
        if args.plotdim:
            if cscale == "log":
                pc = ax.pcolormesh(
                    X,
                    Y,
                    density_tensor,
                    norm="linear",
                    vmax=6,
                    cmap="RdBu_r",
                    linewidth=0,
                    rasterized=True,
                )
                fig.colorbar(pc, ax=ax)
                suffix_n = "cap6"
            else:
                pc = ax.pcolormesh(
                    X,
                    Y,
                    density_tensor,
                    norm="linear",
                    cmap="RdBu_r",
                    linewidth=0,
                    rasterized=True,
                )
                fig.colorbar(pc, ax=ax)
        else:
            pc = ax.pcolormesh(
                X,
                Y,
                density_tensor,
                norm=cscale,
                cmap="RdBu_r",
                linewidth=0,
                rasterized=True,
            )
            fig.colorbar(pc, ax=ax)

        # Disable LaTeX rendering
        plt.rcParams.update({"text.usetex": False})

        ax.set_ylabel("beta", fontsize=14)
        ax.set_xlabel("t", fontsize=14)
        setup = f"step = {args.step}"
        setup += (
            " (norm)"
            if ((args.use_softmax == False) and (args.rawstep == False))
            else ""
        )
        setup += ", softmax=" + ("Y" if args.use_softmax else "N")
        setup += "\n   "
        if args.randomV == 0:
            setup += ", V=Id "
        elif args.randomV == 1:
            setup += ", V=Gsn "
        elif args.randomV == 2:
            setup += ", V=+KQ"
        elif args.randomV == 3:
            setup += ", V=-KQ"

        if args.randomKQ == 0:
            setup += ", KQ=Id "
        elif args.randomKQ == 1:
            setup += ", KQ=Gsn"
        elif args.randomKQ == 2:
            setup += ", KQ=Wig"
        elif args.randomKQ == 3:
            setup += ", KQ=GOE"
        if (args.randomV > 0) or (args.randomKQ > 0):
            if args.noanneal == 1:
                setup += " (no anneal)"
            elif args.noanneal == 2:
                setup += " (single KQV)"

        if args.regen_period > 0:
            setup += f" regen={args.regen_period}"

        # ax.set_title(f'd={args.dmodel}, n={args.ntokens}, ' + setup)
        fig_path = os.path.splitext(fname)[0] + "_" + suffix_n + ".pdf"
        fig.savefig(fig_path, format="pdf", bbox_inches="tight", transparent=True)


# Return histogram of inner products in the 0-th system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dens_sweep")
    # Add device detection at the start
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} backend")
    parser.add_argument(
        "--dmodel", type=int, default=2, help="dimensionality of each token"
    )
    parser.add_argument("--ntokens", type=int, default=32, help="number of tokens")
    parser.add_argument("--batch", type=int, default=2048, help="number of batches")
    parser.add_argument("--betamin", type=float, default=0.1, help="lowest beta")
    parser.add_argument("--betamax", type=float, default=9, help="largest beta")
    parser.add_argument("--betas", type=int, default=90, help="total betas")
    # Good defaults:
    # maxtime=30, softmax=True
    # maxtime=1, softmax=False
    parser.add_argument("--maxtime", type=float, default=0, help="Termination time")
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Time step (for softmax=False it will be renormalized)",
    )
    parser.add_argument(
        "--use_softmax",
        action="store_true",
        help="Use softmax normalization (o/w use ntokens)",
    )
    parser.add_argument(
        "--randomV",
        type=int,
        default=0,
        nargs="?",
        const=1,
        help="V is identity (default), random gsn V (1), +B (2), -B (3)",
    )
    parser.add_argument(
        "--randomKQ",
        type=int,
        default=0,
        nargs="?",
        const=1,
        help="Generate random gsn K,Q (1), or Wigner K^T Q (2)",
    )
    parser.add_argument(
        "--cluster_sizes",
        action="store_true",
        help="Generate random gsn K,Q (o/w identity)",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable TQDM progress bar (for parallel runs)",
    )
    parser.add_argument(
        "--noanneal",
        type=int,
        default=0,
        help="Generate only one K,Q,V per value of beta (1) or per entire run (2)",
    )
    parser.add_argument(
        "--rawstep",
        action="store_true",
        help="For non-softmax, do not normalize the step (fast, but less accurate simu)",
    )
    parser.add_argument(
        "--regen_period",
        type=float,
        default=0,
        help="If non-zero regenerates KQV every <arg> time simulating different layers (default=0)",
    )
    parser.add_argument(
        "--plotdim",
        action="store_true",
        help="Compute dim of span of tokens instead of prob of clustering",
    )
    args = parser.parse_args()
    if args.maxtime == 0.0:
        if args.use_softmax:
            args.maxtime = 30.0
        else:
            args.maxtime = 1.0

    outdict = {
        "args": args,
    }
    salt = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    fname_prefix = datetime.datetime.now().strftime("ds_%Y_%m_%d-%H_%M_" + salt)

    # Create directory for output files
    output_dir = fname_prefix
    os.makedirs(output_dir, exist_ok=True)

    # Update paths to include directory
    fname_prefix = os.path.join(output_dir, fname_prefix)
    args.fname_prefix = fname_prefix
    print(f"Using {fname_prefix}.log for stdout")
    start_time = time.time()

    with open(fname_prefix + ".log", "wt") as sys.stdout:
        print("Using the following settings:\n", args)
        main_res = do_results(args)
        outdict.update(main_res)
        with open(fname_prefix + ".pkl", "wb") as f:
            pickle.dump(outdict, f)

        plot_pickle(fname_prefix + ".pkl")
        end_time = time.time()
        print(f"Total time: {(end_time - start_time) / 60:.1f} min")
