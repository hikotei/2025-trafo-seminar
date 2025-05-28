import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_correlations(hidden_states):
    """
    Compute pairwise correlations between token representations for each layer's hidden states.

    hidden_states (list):
        List of tensors containing hidden states from each layer of the model.
        Each tensor has shape (batch_size=1, sequence_length, hidden_dim)

    Returns list of tensors containing flattened correlation matrices for each layer.
    Each tensor contains the pairwise correlations between all tokens in that layer.
    The correlations are computed as cosine similarities between normalized token representations.
    """
    corrs = []
    for hs in hidden_states:
        # Remove batch dimension and create a copy without gradient tracking
        T = hs.squeeze(0).clone().detach().requires_grad_(False)
        # Normalize each token's representation to unit length for cosine similarity
        T = torch.nn.functional.normalize(T, dim=1)
        # Compute pairwise cosine similarities between all tokens
        T2 = torch.matmul(T, T.transpose(0, 1))
        corrs += [
            T2.flatten().cpu(),  # Flatten matrix and move to CPU for plotting
        ]
    return corrs


def get_random_input(dataset, tokenizer):
    """
    Get a random input sequence from the dataset that meets minimum length requirements.

    dataset: HuggingFace dataset containing text data
    tokenizer: Tokenizer to convert text to model input format

    Q = why only use sequences longer than 300 tokens?
    A = because the model is large and the input sequence needs to be long enough to activate all the parameters

    Returns dictionary containing tokenized input with keys:
        - 'input_ids': Tensor of token IDs
        - 'attention_mask': Tensor of attention masks
    """
    n_train = len(dataset["train"])
    while True:
        # Randomly select an example from the dataset
        it = torch.randint(n_train, (1,)).item()
        text = dataset["train"][it]["text"]
        # Tokenize with padding and truncation
        ei = tokenizer(text, return_tensors="pt", truncation=True)
        # Only use sequences longer than 300 tokens
        if ei["input_ids"].shape[1] > 300:
            break
    return ei


def plot_histograms_save(correls):
    """
    Create and save individual histogram plots for each layer's token correlations.

    Args:
        al_tkz: ALBERT tokenizer instance
        al_model: ALBERT model instance

    The function:
    1. Gets a random input sequence
    2. Computes correlations between tokens for each layer
    3. Creates a separate figure for each layer
    4. Uses adaptive binning based on data distribution
    5. Saves each histogram as a PDF in the 'histograms' directory

    The histograms show the distribution of pairwise correlations between tokens
    in each layer, with consistent y-axis scaling across all plots.
    """

    # Create a directory to save the plots
    os.makedirs("histograms", exist_ok=True)

    # Determine the global maximum density value
    max_density = 0
    for data in correls:
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = max(max_density, max(counts))

    for i, data in enumerate(correls):
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        n = len(data)
        bin_width = 2 * IQR / n ** (1 / 3)
        bins = int((max(data) - min(data)) / bin_width)

        plt.figure()
        plt.hist(
            data,
            bins=bins,
            density=True,
            histtype="step",
            color="#3658bf",
            linewidth=1.5,
        )
        plt.title(f"Layer {i}", fontsize=16)
        plt.xlim(-0.3, 1.05)
        plt.ylim(0, max_density)  # Set a consistent y-axis limit

        plt.savefig(f"./histograms/histogram_layer_{i}.pdf")
        plt.close()
