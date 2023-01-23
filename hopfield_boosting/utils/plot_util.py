from typing import Union, List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

def plot_histograms(X: Union[np.ndarray, List[np.ndarray]],
                    X_labels: Union[None, str, List[str]]=None,
                    bins: Union[int, str, List[float]]='auto',
                    density: bool=False,
                    colormap: Union[str, Colormap, None]=None,
                    colors: Union[None, List[str]]=None) -> plt.figure:
    """
    Plot histograms for single or multiple datasets for visual comparison.

    Parameters:
    - X (Union[np.ndarray, List[np.ndarray]]): Single or list of 1D measurements to be turned into histograms.
    - X_labels (Union[None, str, List[str]], optional): Label or list of labels corresponding to each dataset in X.
    - bins (Union[int, str, List[float]], optional): Specification of histogram bins. Default is 'auto'.
    - density (bool, optional): If True, the histogram represents a probability density. Default is False.
    - colormap (Union[str, Colormap, None], optional): Name of the colormap or a colormap object itself.
    - colors (Union[None, List[str]], optional): List of colors corresponding to each dataset in X.

    Returns:
    - plt.figure: Matplotlib figure object containing the histogram plot.
    """
    if (colormap is not None) and (colors is not None):
        raise ValueError("Both 'colormap' and 'colors' parameters cannot be provided simultaneously. "
                         "Please choose either colormap or colors.")

    if not isinstance(X, list):
        X = [X]
    if isinstance(X_labels, str):
        X_labels = [X_labels]

    if X_labels is not None and len(X) != len(X_labels):
        raise ValueError("Length of X and X_labels should match")

    combined_data = np.concatenate(X)
    common_bins = np.histogram_bin_edges(combined_data, bins=bins)

    if colors is not None:
        if len(colors) != len(X):
            raise ValueError("Length of 'colors' should match the number of datasets in X")
        color_iter = iter(colors)

    if colormap is None:
        colormap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    elif isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    fig, ax = plt.subplots()
    for i, data in enumerate(X):
        label = None if X_labels is None else X_labels[i]
        color = None

        if colors is not None:
            color = next(color_iter)
        else:
            if isinstance(colormap, list):
                color = colormap[i]
            else:
                color = colormap(i / len(X))

        ax.hist(data, bins=common_bins, alpha=0.5, label=label, color=color, density=density)

    ylbl = "Relative Frequency" if density else "Frequency"
    ax.set_ylabel(ylbl)

    if X_labels is not None:
        ax.legend()

    return fig


def create_subplots(n_subplots, rotate=False):
    """
    Creates subplots in a grid layout.

    This function calculates the number of rows and columns based on the desired number of subplots.
    The subplots are then created using the `subplots` function from Matplotlib.

    Parameters:
    n_subplots (int): The total number of subplots.
    rotate (bool): Whether to rotate the layout by 90 degrees.

    Returns:
    matplotlib.figure.Figure: The generated figure object.
    numpy.ndarray: Flattened array of axes objects representing the subplots.

    """
    square_len = np.sqrt(n_subplots)
    # we sometimes need an additional row depending on the rotation and the number of subplots
    row_appendix = int(bool(np.remainder(n_subplots,square_len))*rotate)

    nrows = int(square_len) + row_appendix
    ncols = int((n_subplots+nrows-1) // nrows)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Remove axis and ticks for empty subplots
    for i in range(n_subplots, nrows * ncols):
        axes[i].axis('off')
    
    return fig, axes


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


def ax2ax(source_ax, target_ax):
    """
    Reproduces the contents of one Matplotlib axis onto another axis.

    Parameters:
    source_ax (matplotlib.axes.Axes): The source axis from which the content will be copied.
    target_ax (matplotlib.axes.Axes): The target axis where the content will be reproduced.

    Returns:
    None
    """
    # Reproduce line plots
    for line in source_ax.get_lines():
        target_ax.plot(line.get_xdata(),
                       line.get_ydata(),
                       label=line.get_label(),
                       color=line.get_color(),
                       linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth(),
                       marker=line.get_marker(),
                       markeredgecolor=line.get_markeredgecolor(),
                       markeredgewidth=line.get_markeredgewidth(),
                       markerfacecolor=line.get_markerfacecolor(),
                       markersize=line.get_markersize(),
                      )

    # Reproduce rectangles (histogram bars)
    for artist in source_ax.__dict__['_children']:
        if isinstance(artist, patches.Rectangle):
            rect = artist
            # Retrieve properties of each rectangle and reproduce it on the target axis
            target_ax.add_patch(patches.Rectangle((rect.get_x(), rect.get_y()),
                                                  rect.get_width(),
                                                  rect.get_height(),
                                                  edgecolor=rect.get_edgecolor(),
                                                  facecolor=rect.get_facecolor(),
                                                  linewidth=rect.get_linewidth(),
                                                  linestyle=rect.get_linestyle()
                                                 ))

    # Reproduce collections (e.g., LineCollection)
    for collection in source_ax.collections:
        if isinstance(collection, plt.collections.LineCollection):
            lc = plt.collections.LineCollection(segments=collection.get_segments(),
                                                label=collection.get_label(),
                                                color=collection.get_color(),
                                                linestyle=collection.get_linestyle(),
                                                linewidth=collection.get_linewidth(),
                                               )
            target_ax.add_collection(lc)

    # Reproduce axis limits and aspect ratio
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())
    target_ax.set_aspect(source_ax.get_aspect())

    # Reproduce axis labels
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    
    # Reproduce title
    target_ax.set_title(source_ax.get_title())

    # Reproduce legend
    handles, labels = source_ax.get_legend_handles_labels()
    target_ax.legend(handles, labels)


def plot_random_samples(dataset: torch.utils.data.Dataset, n_samples: int,
                        grid_layout: Optional[bool] = True, title_label: Optional[bool] = True,
                        frame: Optional[bool] = True) -> plt.Figure:
    """
    Plot random samples from a PyTorch dataset.

    Parameters:
        dataset (torch.utils.data.Dataset): PyTorch dataset containing images and labels.
        n_samples (int): Number of samples to plot.
        grid_layout (bool, optional): If True, arrange the samples in a grid layout. Default is True.
        title_label (bool, optional): If True, display labels as titles for each sample. Default is True.
        frame (bool, optional): If True, display frames around the images. Default is True.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure containing the plotted samples.
    """
    # Randomly sample indices
    n_images = len(dataset)
    #sampled_indices = torch.randperm(n_images)[:n_samples]
    sampled_indices = np.random.permutation(n_images)[:n_samples]

    # Extract images and labels
    sampled_images = [dataset[i][0] for i in sampled_indices]
    if title_label:
        sampled_labels = [dataset[i][1] for i in sampled_indices]

    if grid_layout:
        # Create the layout of subplots
        fig, axes = create_subplots(n_samples, rotate=False)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=n_samples)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(np.transpose(sampled_images[i], (1, 2, 0)))

        if title_label:
            ax.set_title(f"Label: {sampled_labels[i]}")

        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        if not frame:
            ax.axis("off")

    return fig


def plot_random_samples_multi(datasets: Union[torch.utils.data.Dataset, List[torch.utils.data.Dataset]],
                               n_samples: int, title_label: Optional[bool] = True,
                               frame: Optional[bool] = True) -> plt.Figure:
    """
    Plot random samples from one or more PyTorch datasets.

    Parameters:
        datasets (Union[torch.utils.data.Dataset, List[torch.utils.data.Dataset]]): Single or list of PyTorch datasets.
        n_samples (int): Number of samples to plot for each dataset.
        title_label (bool, optional): If True, display labels as titles for each sample. Default is True.
        frame (bool, optional): If True, display frames around the images. Default is True.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure containing the plotted samples.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]

    n_datasets = len(datasets)

    fig, axes = plt.subplots(nrows=n_datasets, ncols=n_samples)

    if n_datasets < 2:
        axes = np.expand_dims(axes, axis=0)  # Add an extra dimension for consistency

    for dataset_index, dataset in enumerate(datasets):
        # Randomly sample indices
        n_images = len(dataset)
        sampled_indices = torch.randperm(n_images)[:n_samples]

        # Extract images and labels
        sampled_images = [dataset[i][0] for i in sampled_indices]
        if title_label:
            sampled_labels = [dataset[i][1] for i in sampled_indices]

        for i in range(n_samples):
            ax = axes[dataset_index, i]
            ax.imshow(np.transpose(sampled_images[i], (1, 2, 0)))

            if title_label:
                ax.set_title(f"Label: {sampled_labels[i]}")

            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            if not frame:
                ax.axis("off")

    return fig    