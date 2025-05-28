# %%
import os
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset # Hugging Face Datasets
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig

# %%
# Load the WikiText-103 dataset (version 1)
wikitext = load_dataset("wikitext", "wikitext-103-v1")

# %% [markdown]
# ### Normal Model (24 layers)

# %%
al_tkz = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
al_model = AlbertModel.from_pretrained("albert-xlarge-v2")

# %%
def compute_correlations(hidden_states):
    """
    Compute pairwise correlations between token representations for each layer's hidden states.

    hidden_states (list): List of tensors containing hidden states from each layer of the model.
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

# %%
def get_random_input(dataset, tokenizer):
    """
    Get a random input sequence from the dataset that meets minimum length requirements.
    
    Args:
        dataset: HuggingFace dataset containing text data
        tokenizer: Tokenizer to convert text to model input format
    
    Returns:
        dict: Dictionary containing tokenized input with keys:
            - 'input_ids': Tensor of token IDs
            - 'attention_mask': Tensor of attention masks
            The sequence length will be > 300 tokens
    """
    l = len(dataset["train"])
    while True:
        # Randomly select an example from the dataset
        it = torch.randint(l, (1,)).item()
        text = dataset["train"][it]["text"]
        # Tokenize with padding and truncation
        ei = tokenizer(text, return_tensors="pt", truncation=True)
        # Only use sequences longer than 300 tokens
        if ei["input_ids"].shape[1] > 300:
            break
    return ei

# %%
def plot_histograms(al_tkz=al_tkz, al_model=al_model):
    """
    Plot histograms of token correlations for all layers in a single figure.
    
    Args:
        al_tkz: ALBERT tokenizer instance
        al_model: ALBERT model instance
    
    The function:
    1. Gets a random input sequence
    2. Computes correlations between tokens for each layer
    3. Creates a grid of histograms (5x5 for 24-layer model, 7x7 for 48-layer model)
    4. Plots correlation distributions for each layer
    """
    ei = get_random_input(wikitext, al_tkz)
    print(al_tkz.batch_decode(ei["input_ids"]))
    # Get model outputs including all hidden states
    of = al_model(**ei, output_hidden_states=True)
    correls = compute_correlations(of["hidden_states"])
    # Create appropriate grid size based on number of layers
    if al_model.config.num_hidden_layers < 25:
        fig, axes = plt.subplots(5, 5)
    else:
        fig, axes = plt.subplots(7, 7)
    axes = axes.flatten()
    for i in range(len(correls)):
        # Plot correlation distribution for each layer
        axes[i].hist(correls[i], bins=100, density=True, histtype="step")
        axes[i].set_title(f"Layer {i}")
        axes[i].set_xlim(-0.3, 1)  # Set x-axis limits for correlation values

# %%
def plot_histograms_save(al_tkz=al_tkz, al_model=al_model):
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
    ei = get_random_input(wikitext, al_tkz)

    print('- '*20)
    print("Random Input:")
    print(al_tkz.batch_decode(ei["input_ids"]))
    print('- '*20)
    
    of = al_model(**ei, output_hidden_states=True)
    correls = compute_correlations(of["hidden_states"])

    # Create a directory to save the plots
    os.makedirs("histograms", exist_ok=True)

    # Find maximum density across all layers for consistent y-axis scaling
    max_density = 0
    for data in correls:
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = max(max_density, max(counts))

    for i, data in enumerate(correls):
        # Calculate optimal bin width using Freedman-Diaconis rule
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
        plt.xlim(-0.3, 1.05)  # Set x-axis limits for correlation values
        plt.ylim(0, max_density)  # Set consistent y-axis limit across all plots

        plt.savefig(f"./histograms/histogram_layer_{i}.pdf")
        plt.close()


# %%
plot_histograms_save()

# %%
# al_model.config

# %%
## Check norm of output tokens
# The norm is not exactly the same because the LayerNorm
# that is applied at the end also has trainable diagonal matrix \gamma and vector \beta which are used as follows
# (on each token)
# \tilde x = (x - mean(x))/sqrt(var(x)) * \gamma + \beta (here token is a row vector)

# ei = get_random_input(wikitext, al_tkz)
# print(al_tkz.batch_decode(ei["input_ids"]))
# of = al_model(**ei, output_hidden_states=True)
# of["hidden_states"][24].var(2)

# %% [markdown]
# ### Larger Model (48 layers)

# %%
alm2_config = AlbertConfig.from_pretrained(
    "albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1
)
almodel2 = AlbertModel.from_pretrained("albert-xlarge-v2", config=alm2_config)
print(almodel2.config.num_hidden_layers)

# %%
# %matplotlib inline
plot_histograms_save(al_model=almodel2)
# plot_histograms(al_model = almodel2)

# %% [markdown]
# ### Decomposing Internal Structure

# %% [markdown]
# This section appears to be a debugging/exploration section where the author is:
# 
# - Understanding the internal structure of the ALBERT model
# - Testing how the attention mechanism processes inputs
# - Examining the shapes and transformations of tensors through the model
# - Looking at the actual weight matrices used in the attention mechanism
# 
# This kind of exploration is common when trying to understand or debug transformer models, as it helps to verify that the internal components are working as expected and to understand how the model processes information at a detailed level.

# %%
# Decomposing AlbertModel
al_transfo = al_model.encoder
al_layer = al_transfo.albert_layer_groups[0].albert_layers[0]
al_attention = al_layer.attention
# Print attention mechanism configuration
(
    al_attention.pruned_heads,  # Which attention heads have been pruned
    al_attention.num_attention_heads,  # Number of attention heads
    al_attention.all_head_size,  # Size of each attention head
    al_attention.hidden_size,  # Total hidden size of the model
)

# %%
ei = get_random_input(wikitext, al_tkz)
print(al_tkz.batch_decode(ei["input_ids"]))
of = al_model(**ei, output_hidden_states=True);

# %%
# Check shape of hidden states from layer 2
of["hidden_states"][2].shape

# %%
# Create test tensor to examine attention mechanism
test = torch.arange(2048).reshape(1, 1, -1)
print(test.shape)
# Check how attention mechanism reshapes the input
al_attention.transpose_for_scores(test).shape;

# %%
# Examine the first attention head's transformation of the first token
al_attention.transpose_for_scores(test)[0, 1, 0, :]

# %%
# Look at the value projection weights of the attention mechanism
al_attention.value.weight


