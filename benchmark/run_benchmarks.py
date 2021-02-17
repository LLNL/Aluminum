"""Run all (or a subset of) Aluminum benchmarks."""

import argparse
import subprocess
import math
import os.path
from collections import namedtuple

parser = argparse.ArgumentParser(
    description='Run all Aluminum benchmarks')
parser.add_argument('--out-dir', type=str, required=True,
                    help='Directory to save results in')
parser.add_argument('--num-nodes', type=int, required=True,
                    help='Max number of nodes to use')
parser.add_argument('--procs-per-node', type=int, default=4,
                    help='Number of processes per node')
parser.add_argument('--min-procs', type=int,
                    help='Minimum number of processes to use')
parser.add_argument('--launcher', type=str, default='jsrun',
                    choices=['jsrun', 'srun'],
                    help='Which parallel launcher to use')
parser.add_argument('--benchmark-ops', type=str, default='./benchmark_ops.exe',
                    help='Path to benchmark_ops binary')
parser.add_argument('--extra-args', type=str,
                    help='Extra arguments to pass to launcher')
parser.add_argument('--benchmarks', type=str, nargs='+',
                    help='Specify particular benchmarks to run')
parser.add_argument('--datatypes', type=str, nargs='+',
                    help='Specify particular datatypes to use')
parser.add_argument('--backends', type=str, nargs='+',
                    choices=['mpi', 'nccl', 'ht'],
                    help='Specify which backends to use')
parser.add_argument('--inplace', default=None, action='store_true',
                    help='Run only inplace algorithms')
parser.add_argument('--notinplace', default=None, action='store_true',
                    help='Run only out-of-place algorithms')
parser.add_argument('--blocking', default=None, action='store_true',
                    help='Run only blocking algorithms')
parser.add_argument('--nonblocking', default=None, action='store_true',
                    help='Run only nonblocking algorithms')


# TODO: This is copied from run_tests.py, I'd prefer to have a single source.
# Supported datatypes for backends.
mpi_datatypes = ['char', 'schar', 'uchar', 'short', 'ushort', 'int', 'uint',
                 'long', 'ulong', 'longlong', 'ulonglong',
                 'float', 'double', 'longdouble']
nccl_datatypes = ['char', 'uchar', 'int', 'uint', 'longlong', 'ulonglong',
                  'half', 'float', 'double']
# Standard sets of operations.
# inplace is one of 'both', True, or False.
# root is either True or False.
# algorithms is a dict mapping backends to algorithm names.
_default_algo_map = {
    'mpi': ['automatic'],
    'nccl': ['automatic'],
    'ht': ['automatic']
}
OpDesc = namedtuple('OpDesc',
                    ['op', 'inplace', 'root', 'min_procs', 'algorithms'],
                    defaults=['both', False, 1, _default_algo_map])
coll_ops = [OpDesc('allgather'),
            OpDesc('allreduce',
                   algorithms={'mpi': ['automatic', 'passthrough',
                                       'recursive_doubling', 'ring',
                                       'rabenseifner'],  # Excluding biring.
                               'nccl': ['automatic'],
                               'ht': ['automatic']}),
            OpDesc('alltoall'),
            OpDesc('bcast', inplace=True, root=True),
            OpDesc('gather', root=True),
            OpDesc('reduce', root=True),
            OpDesc('reduce_scatter'),
            OpDesc('scatter', root=True)]
vector_coll_ops = [OpDesc('allgatherv'),
                   OpDesc('alltoallv'),
                   OpDesc('gatherv', root=True),
                   OpDesc('reduce_scatterv'),
                   OpDesc('scatterv', root=True)]
pt2pt_ops = [OpDesc('send', inplace=False, min_procs=2),
             OpDesc('recv', inplace=False, min_procs=2),
             OpDesc('sendrecv', inplace=False, min_procs=2)]

# Full set of cases.
test_cases = {
    'mpi': {
        'ops': coll_ops + vector_coll_ops + pt2pt_ops,
        'datatypes': mpi_datatypes
    },
    'nccl': {
        'ops': coll_ops + vector_coll_ops + pt2pt_ops,
        'datatypes': nccl_datatypes
    },
    'ht': {
        'ops': coll_ops + vector_coll_ops + pt2pt_ops,
        'datatypes': mpi_datatypes
    },
}


def get_ppn_and_nodes(num_procs, procs_per_node):
    """Return the number of nodes and processors per node to use."""
    if num_procs >= procs_per_node:
        ppn = procs_per_node
        num_nodes = num_procs // ppn
    else:
        ppn = num_procs
        num_nodes = 1
    return ppn, num_nodes


def get_jsrun_launcher(num_procs, args):
    """Return the base launch command using jsrun."""
    ppn, num_nodes = get_ppn_and_nodes(num_procs, args.procs_per_node)
    return ['jsrun',
            '--nrs', str(num_nodes),
            '--rs_per_host', '1',
            '--tasks_per_rs', str(ppn),
            '--cpu_per_rs', 'ALL_CPUS',
            '--gpu_per_rs', 'ALL_GPUS',
            '--bind', 'packed:8',
            '--launch_distribution', 'packed']


def get_srun_launcher(num_procs, args):
    """Return the base launch command using srun."""
    ppn, num_nodes = get_ppn_and_nodes(num_procs, args.procs_per_node)
    return ['srun',
            f'--nodes={num_nodes}',
            f'--ntasks-per-node={ppn}',
            '--mpibind=off',
            '--nvidia_compute_mode=default']


launcher_funcs = {
    'jsrun': get_jsrun_launcher,
    'srun': get_srun_launcher
}

def run_benchmark(args, num_procs, backend, operator, datatype, algorithm,
                  inplace, nonblocking, root):
    """Run a specified benchmark."""
    out_file = (f'{operator}_{backend}_{datatype}_{algorithm}_{num_procs}'
                f'_{"inplace" if inplace else "notinplace"}'
                f'_{"nonblocking" if nonblocking else "blocking"}')
    if root is not None:
        out_file += f'_root{root}'
    out_path = os.path.join(args.out_dir, out_file)
    if os.path.exists(out_path):
        raise RuntimeError('Output file ' + out_path + ' exists')
    launcher_cmd = launcher_funcs[args.launcher](num_procs, args)
    if args.extra_args:
        launcher_cmd += args.extra_args.split(' ')
    test_cmd = [args.benchmark_ops,
                '--op', operator,
                '--backend', backend,
                '--datatype', datatype,
                '--algorithm', algorithm,
                '--save-to-file', out_path,
                '--no-print-table']
    test_desc = f'procs:{num_procs} {backend} {operator} {datatype} {algorithm}'
    if inplace:
        test_cmd += ['--inplace']
        test_desc += ' inplace'
    if nonblocking:
        test_cmd += ['--nonblocking']
        test_desc += ' nonblocking'
    if root is not None:
        test_cmd += ['--root', str(root)]
        test_desc += f' root:{root}'
    r = subprocess.run(launcher_cmd + test_cmd, capture_output=True, text=True,
                       check=False)
    if r.returncode == 0:
        print('[Done] ' + test_desc)
    else:
        print('[Fail] ' + test_desc)
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr)


def run_all_benchmarks(args):
    """Run the full (or specified) benchmark suite."""
    os.makedirs(args.out_dir, exist_ok=True)
    procs = [2**x for x in range(
        int(math.log2(args.num_nodes * args.procs_per_node)) + 1)]
    if args.min_procs:
        procs = list(filter(lambda x: x >= args.min_procs, procs))
    for num_procs in procs:
        for backend, cases in test_cases.items():
            if args.backends and backend not in args.backends:
                continue
            for opdesc in cases['ops']:
                if args.benchmarks and opdesc.op not in args.benchmarks:
                    continue
                if num_procs < opdesc.min_procs:
                    continue
                for datatype in cases['datatypes']:
                    if args.datatypes and datatype not in args.datatypes:
                        continue
                    blocking_cases = []
                    if args.blocking is not None:
                        blocking_cases.append(False)
                    if args.nonblocking is not None:
                        blocking_cases.append(True)
                    if args.blocking is None and args.nonblocking is None:
                        blocking_cases = [False, True]
                    for nonblocking in blocking_cases:
                        # If operator only supports one mode, always use it.
                        if opdesc.inplace == 'both':
                            inplace_cases = []
                            if args.inplace:
                                inplace_cases.append(True)
                            if args.notinplace:
                                inplace_cases.append(False)
                            if not args.inplace and not args.notinplace:
                                inplace_cases = [True, False]
                        else:
                            inplace_cases = [opdesc.inplace]
                        for inplace in inplace_cases:
                            root = 0 if opdesc.root else None
                            for algorithm in opdesc.algorithms[backend]:
                                run_benchmark(
                                    args, num_procs, backend, opdesc.op,
                                    datatype, algorithm,
                                    inplace, nonblocking, root)


if __name__ == '__main__':
    run_all_benchmarks(parser.parse_args())
