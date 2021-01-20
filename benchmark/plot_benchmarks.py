"""Plot benchmark results.

Currently plotted figures:
- For a fixed operator and process count, time versus message size.
- For a fixed operator and message size, time versus process count.
- For a fixed operator, plot a process count x message size heatmap
showing the fastest implementation.

Data is either aggregated together over ranks or plotted separately
for each rank. Rooted collectives default to plotting the root and
non-root ranks separately.

"""

import argparse
import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set()


parser = argparse.ArgumentParser(
    description='Plot results from Aluminum benchmarks')
parser.add_argument('results', type=str, nargs='+',
                    help='Directories containing results to plot')
parser.add_argument('--out-dir', type=str, required=True,
                    help='Directory to save plots to')
parser.add_argument('--out-of-place', default=False, action='store_true',
                    help='Plot results for not-inplace runs')
parser.add_argument('--nonblocking', default=False, action='store_true',
                    help='Plot results for nonblocking runs')
parser.add_argument('--datatype', type=str, default=None,
                    help='Filter to a particular datatype')
parser.add_argument('--backend', type=str, default=None, nargs='+',
                    help='Filter to particular backends')
parser.add_argument('--op', type=str, default=None, nargs='+',
                    help='Filter to particular operations')
parser.add_argument('--min-procs', type=int, default=None,
                    help='Minimum number of processes to plot')
parser.add_argument('--max-procs', type=int, default=None,
                    help='Maximum number of processes to plot')
parser.add_argument('--min-size', type=int, default=None,
                    help='Minimum message size to plot')
parser.add_argument('--max-size', type=int, default=None,
                    help='Maximum message size to plot')
parser.add_argument('--run-names', type=str, default=None, nargs='+',
                    help='Special names to give each run')
parser.add_argument('--separate-ranks', default=False, action='store_true',
                    help='Make a separate plot for each rank')


rooted_ops = set([
    'bcast', 'gather', 'gatherv', 'reduce', 'scatter', 'scatterv'])
pt2pt_ops = set(['send', 'recv', 'sendrecv'])


def load_result_directory(result_dir, inplace, nonblocking):
    """Load all data in result_dir.

    A new column, Run, will be added to the dataframe with the value
    being result_dir to identify it.

    inplace is True to keep inplace results, False for out-of-place.
    nonblocking is True to keep nonblocking results, False for blocking.

    Returns separate dataframes for collective and point-to-point ops.

    """
    coll_results = []
    pt2pt_results = []
    for filename in os.listdir(result_dir):
        try:
            df = pd.read_csv(os.path.join(result_dir, filename),
                             delim_whitespace=True)
        except:
            print(('Could not load benchmark data from '
                   f'{os.path.join(result_dir, filename)}'))
            continue
        op = df.Operation[0]
        if op in pt2pt_ops:
            pt2pt_results.append(df)
        else:
            coll_results.append(df)
    if not coll_results and not pt2pt_results:
        raise RuntimeError(f'No results in {result_dir} could be loaded')
    if coll_results:
        coll_df = pd.concat(coll_results)
    else:
        coll_df = pd.DataFrame(columns=[
            'Backend', 'Type', 'Operation', 'Algo', 'NonBlocking', 'InPlace',
            'Root', 'CommSize', 'Size', 'CommRank', 'Time'])
    if pt2pt_results:
        pt2pt_df = pd.concat(pt2pt_results)
    else:
        pt2pt_df = pd.DataFrame(columns=[
            'Backend', 'Type', 'Operation', 'CommSize', 'Size', 'CommRank',
            'Time'])
    if inplace:
        coll_df = coll_df[coll_df.InPlace == 1]
    else:
        coll_df = coll_df[coll_df.InPlace == 0]
    if nonblocking:
        coll_df = coll_df[coll_df.NonBlocking == 1]
    else:
        coll_df = coll_df[coll_df.NonBlocking == 0]
    # No longer need these columns.
    coll_df.drop(columns=['InPlace', 'NonBlocking'], inplace=True)
    return coll_df, pt2pt_df


def filter_df(df, args):
    """Filter entries in df according to arguments.

    The nonblocking and inplace arguments are handled at loading.

    """
    if args.datatype:
        df = df[df.Type == args.datatype]
    if args.backend:
        restrict = False
        for backend in args.backend:
            restrict |= df.Backend == backend
        df = df[restrict]
    if args.op:
        restrict = False
        for op in args.op:
            restrict |= df.Operation == op
        df = df[restrict]
    if args.min_procs is not None:
        df = df[df.CommSize >= args.min_procs]
    if args.max_procs is not None:
        df = df[df.CommSize <= args.max_procs]
    if args.min_size is not None:
        df = df[df.Size >= args.min_size]
    if args.max_size is not None:
        df = df[df.Size <= args.max_size]
    return df


def get_config_name(op, args, rank=None, root=None):
    """Return a base name for a configuration."""
    name = op
    if args.out_of_place:
        name += '_notinplace'
    else:
        name += '_inplace'
    if args.nonblocking:
        name += '_nonblocking'
    else:
        name += '_blocking'
    if rank:
        name += f'_rank{rank}'
    if root is True:
        name += '_root'
    elif root is False:
        name += '_nonroot'
    return name


def get_figsize(df, x, hue):
    """Attempt to get a reasonable figure size."""
    num_x = len(df[x].unique())
    num_hue = len(df[hue].unique())
    width_per_vplot = 0.3
    width_per_tick = width_per_vplot * num_hue
    width = num_x*width_per_tick + 2  # Extra space for axes/etc.
    # Just using default value for y for now.
    return width, 4.8


def estimate_reasonable_max(df, x):
    """Estimate a reasonable maximum value for the plot y axis."""
    # Use 1.5 IQR of the largest group.
    group = df.groupby(x)
    q1 = group.Time.quantile(0.25)
    q3 = group.Time.quantile(0.75)
    iqr = (q3 - q1).max()
    return q3.max() + 1.5*iqr


def plot_time_v_procs(df, out_dir, name):
    """Plot time vs process count for a fixed operator."""
    for size, size_df in df.groupby('Size'):
        fig, ax = plt.subplots(figsize=get_figsize(size_df, 'CommSize', 'Impl'))
        sns.violinplot(x='CommSize', y='Time', hue='Impl', data=size_df,
                       cut=0, inner='quartile', linewidth=0.25,
                       ax=ax)
        ymax = min(estimate_reasonable_max(size_df, 'CommSize'),
                   ax.get_ylim()[1])
        ax.set_ylim(bottom=0, top=ymax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical',
                           fontsize='small')
        ax.legend(fontsize='x-small')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + f'_tvp_size{size}.pdf'))
        plt.close()


def plot_time_v_sizes(df, out_dir, name):
    """Plot time vs message size for a fixed operator."""
    for procs, proc_df in df.groupby('CommSize'):
        fig, ax = plt.subplots(figsize=get_figsize(proc_df, 'Size', 'Impl'))
        sns.violinplot(x='Size', y='Time', hue='Impl', data=proc_df,
                       cut=0, inner='quartile', linewidth=0.25,
                       ax=ax)
        ax.set_yscale('log')
        ymax = min(estimate_reasonable_max(proc_df, 'Size'),
                   ax.get_ylim()[1])
        ax.set_ylim(top=ymax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical',
                           fontsize='small')
        ax.legend(fontsize='x-small')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + f'_tvs_procs{procs}.pdf'))
        plt.close()


def plot_best_heatmap(df, out_dir, name):
    """Plot the best-performing implementation for each processor/size
    combination."""
    # Compute medians for each implementation / processor / size group.
    df_median = df.groupby(['Impl', 'CommSize', 'Size'], as_index=False).Time.median()
    # Now get rows with winning implementations.
    df_best = df_median.loc[df_median.groupby(['CommSize', 'Size']).Time.idxmin()].copy()
    # Get numerical indices for implementations.
    impls = list(df.Impl.unique())
    num_impls = len(impls)
    impl_indices = dict(zip(impls, range(num_impls)))
    # Also map sizes to indices.
    comm_sizes = sorted(df.CommSize.unique())
    comm_size_indices = dict(zip(comm_sizes, range(len(comm_sizes))))
    sizes = sorted(df.Size.unique())
    size_indices = dict(zip(sizes, range(len(sizes))))
    # Remap all values.
    df_best.Impl = df_best.Impl.map(impl_indices)
    df_best.CommSize = df_best.CommSize.map(comm_size_indices)
    df_best.Size = df_best.Size.map(size_indices)
    # Build the matrix for the heatmap.
    # Not sure of a better way than iterating.
    best = np.zeros((len(sizes), len(comm_sizes)), dtype=np.int)
    for r in df_best.iterrows():
        r = r[1]
        best[int(r.Size), int(r.CommSize)] = r.Impl
    # Plot heatmap.
    fig, ax = plt.subplots()
    cmap = sns.color_palette(n_colors=num_impls)
    sns.heatmap(best, cmap=cmap, xticklabels=comm_sizes, yticklabels=sizes,
                vmin=0, vmax=num_impls-1,
                ax=ax)
    ax.set_xlabel('Processes')
    ax.set_ylabel('Message size')
    # Add labels to color bar.
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks(
        [colorbar.vmin + r/num_impls*(0.5+i) for i in range(num_impls)])
    colorbar.set_ticklabels(impls)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + '_best.pdf'))
    plt.close()


def do_all_plots(df, args, name):
    """Generate all plots for a dataframe."""
    plot_time_v_procs(df, args.out_dir, name)
    plot_time_v_sizes(df, args.out_dir, name)
    plot_best_heatmap(df, args.out_dir, name)


def plot_all_results(args):
    """Generate all plots for results."""
    multiple_results = len(args.results) > 1
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.run_names:
        args.run_names = args.results
    if len(args.run_names) != len(args.results):
        raise ValueError('If using --run-names, must name every run')

    coll_dfs = []
    pt2pt_dfs = []
    for result_dir, run_name in zip(args.results, args.run_names):
        coll_df, pt2pt_df = load_result_directory(
            result_dir, not args.out_of_place, args.nonblocking)
        # Create a column to identify run/backend/algorithm combinations.
        if multiple_results:
            coll_df['Impl'] = (run_name + '_' + coll_df['Backend']
                               + '_' + coll_df['Algo'])
            pt2pt_df['Impl'] = run_name + '_' + pt2pt_df['Backend']
        else:
            # Don't bother with the run here.
            coll_df['Impl'] = coll_df['Backend'] + '_' + coll_df['Algo']
            pt2pt_df['Impl'] = pt2pt_df['Backend']
        if not coll_df.empty:
            coll_dfs.append(coll_df)
        if not pt2pt_df.empty:
            pt2pt_dfs.append(pt2pt_df)
    if coll_dfs:
        coll_df = filter_df(pd.concat(coll_dfs), args)
        for op, op_df in coll_df.groupby('Operation'):
            if args.separate_ranks:
                for rank, rank_op_df in op_df.groupby('CommRank'):
                    do_all_plots(rank_op_df, args,
                                 get_config_name(op, args, rank=rank))
            elif op in rooted_ops:
                root_op_df = op_df[op_df.CommRank == op_df.Root]
                do_all_plots(root_op_df, args,
                             get_config_name(op, args, root=True))
                notroot_op_df = op_df[op_df.CommRank != op_df.Root]
                do_all_plots(notroot_op_df, args,
                             get_config_name(op, args, root=False))
            else:
                do_all_plots(op_df, args, get_config_name(op, args))
    if pt2pt_dfs:
        pt2pt_df = filter_df(pd.concat(pt2pt_dfs), args)
        for op, op_df in pt2pt_df.groupby('Operation'):
            if args.separate_ranks:
                for rank, rank_op_df in op_df.groupby('CommRank'):
                    do_all_plots(rank_op_df, args,
                                 get_config_name(op, args, rank=rank))
            else:
                do_all_plots(op_df, args, get_config_name(op, args))


if __name__ == '__main__':
    plot_all_results(parser.parse_args())
