"""Run all Aluminum tests."""

import argparse
import subprocess
from collections import namedtuple
import math


parser = argparse.ArgumentParser(
    description='Run all Aluminum tests')
parser.add_argument('--num-nodes', type=int, required=True,
                    help='Max number of nodes to use')
parser.add_argument('--procs-per-node', type=int, default=4,
                    help='Number of processes per node')
parser.add_argument('--launcher', type=str, default='jsrun',
                    choices=['jsrun', 'srun'],
                    help='Which parallel launcher to use')
parser.add_argument('--test-ops', type=str, default='./test_ops.exe',
                    help='Path to test_ops binary')
parser.add_argument('--extra-args', type=str,
                    help='Extra arguments to pass to launcher')


# Supported datatypes for backends.
mpi_datatypes = ['char', 'schar', 'uchar', 'short', 'ushort', 'int', 'uint',
                 'long', 'ulong', 'longlong', 'ulonglong',
                 'float', 'double', 'longdouble']
nccl_datatypes = ['char', 'uchar', 'int', 'uint', 'longlong', 'ulonglong',
                  'half', 'float', 'double']
# Standard sets of operations.
# inplace is one of 'both', True, or False.
# root is either True or False.
OpDesc = namedtuple('OpDesc', ['op', 'inplace', 'root', 'min_procs'],
                    defaults=['both', False, 1])
coll_ops = [OpDesc('allgather'),
            OpDesc('allreduce'),
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
        'ops': coll_ops + pt2pt_ops,
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


def run_test(args, num_procs, backend, operator, datatype, inplace,
             nonblocking, root):
    """Run a specified test."""
    launcher_cmd = launcher_funcs[args.launcher](num_procs, args)
    if args.extra_args:
        launcher_cmd += args.extra_args.split(' ')
    test_cmd = [args.test_ops,
                '--op', operator,
                '--backend', backend,
                '--datatype', datatype,
                # Keep things relatively small.
                '--max-size', '2048',
                # Don't wait too long.
                '--hang-timeout', '5']
    test_desc = f'procs:{num_procs} {backend} {operator} {datatype}'
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
        print('[Pass] ' + test_desc)
    else:
        print('[Fail] ' + test_desc)
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr)


def run_all_tests(args):
    """Run the full test suite."""
    procs = [2**x for x in range(
        int(math.log2(args.num_nodes * args.procs_per_node)) + 1)]
    for num_procs in procs:
        for backend, cases in test_cases.items():
            for opdesc in cases['ops']:
                if num_procs < opdesc.min_procs:
                    continue
                for datatype in cases['datatypes']:
                    for nonblocking in [True, False]:
                        inplace_cases = [True, False] if opdesc.inplace == 'both' else [opdesc.inplace]
                        for inplace in inplace_cases:
                            root_cases = [0] if opdesc.root else [None]
                            if num_procs > 1 and opdesc.root:
                                root_cases += [1]
                            for root in root_cases:
                                run_test(args, num_procs, backend, opdesc.op,
                                         datatype, inplace, nonblocking, root)


if __name__ == '__main__':
    run_all_tests(parser.parse_args())
