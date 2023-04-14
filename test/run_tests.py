"""Run all Aluminum tests."""

import argparse
import subprocess
from collections import namedtuple
import math
import os.path


parser = argparse.ArgumentParser(
    description='Run all Aluminum tests')
parser.add_argument('--num-nodes', type=int, required=True,
                    help='Max number of nodes to use')
parser.add_argument('--procs-per-node', type=int, default=4,
                    help='Number of processes per node')
parser.add_argument('--min-procs', type=int,
                    help='Minimum number of processes to use')
parser.add_argument('--launcher', type=str, default='jsrun',
                    choices=['jsrun', 'srun', 'flux'],
                    help='Which parallel launcher to use')
parser.add_argument('--test-ops', type=str, default='./test_ops',
                    help='Path to test_ops binary')
parser.add_argument('--extra-args', type=str,
                    help='Extra arguments to pass to launcher')
parser.add_argument('--operations', type=str, nargs='+',
                    help='Specify particular operations to run')
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
parser.add_argument('--threads', type=int, default=None,
                    help='Number of threads to test with')
parser.add_argument('--no-abort-on-hang', default=None, action='store_true',
                    help='Do not abort when a hang is detected')


# Default time spent waiting before declaring the process hung (in seconds)
hang_timeout = 5
# Supported datatypes for backends.
mpi_datatypes = ['char', 'schar', 'uchar', 'short', 'ushort', 'int', 'uint',
                 'long', 'ulong', 'longlong', 'ulonglong',
                 'float', 'double', 'half', 'bfloat16']# 'longdouble']
nccl_datatypes = ['char', 'uchar', 'int', 'uint', 'longlong', 'ulonglong',
                  'float', 'double', 'half', 'bfloat16']
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
             OpDesc('sendrecv', inplace='both', min_procs=2)]

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


def get_jsrun_launcher(num_procs, args, ppn, num_nodes):
    """Return the base launch command using jsrun."""
    return ['jsrun',
            '--nrs', str(num_nodes),
            '--rs_per_host', '1',
            '--tasks_per_rs', str(ppn),
            '--cpu_per_rs', 'ALL_CPUS',
            '--gpu_per_rs', 'ALL_GPUS',
            '--bind', 'packed:8',
            '--launch_distribution', 'packed']


def get_srun_launcher(num_procs, args, ppn, num_nodes):
    """Return the base launch command using srun."""
    return ['srun',
            f'-n {num_nodes * ppn}',
            f'--nodes={num_nodes}',
            f'--ntasks-per-node={ppn}',
            '--mpibind=off']
            #'--nvidia_compute_mode=default']


def get_flux_launcher(num_procs, args, ppn, num_nodes):
    """Return the base launch command using flux."""
    return ['flux', 'mini', 'run',
            f'--nodes={num_nodes}',
            f'--tasks-per-node={ppn}',
            '-o', 'mpibind=off']


launcher_funcs = {
    'jsrun': get_jsrun_launcher,
    'srun': get_srun_launcher,
    'flux': get_flux_launcher
}


def get_launcher_cmd(num_procs, args, ppn=None, num_nodes=None):
    """Return the launcher command."""
    default_ppn, default_num_nodes = get_ppn_and_nodes(num_procs,
                                                       args.procs_per_node)
    if ppn is None:
        ppn = default_ppn
    if num_nodes is None:
        num_nodes = default_num_nodes
    return launcher_funcs[args.launcher](num_procs, args, ppn, num_nodes)


def run_subprocess(process_args, timeout=None):
    """Run a subprocess.

    This supports a timeout without killing the process.

    """
    if timeout is None:
        # Just use the standard interface in this case.
        return subprocess.run(process_args, capture_output=True, text=True,
                              check=False)
    with subprocess.Popen(process_args,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True) as process:
        stdout = ''
        stderr = ''
        while True:
            try:
                r = process.communicate(None, timeout=timeout)
                stdout += r[0]
                stderr += r[1]
                break
            except subprocess.TimeoutExpired as e:
                # Ran out of time. Report what we got and keep waiting.
                print('Hit timeout in process',
                      ' '.join(process_args), flush=True)
                if e.stdout:
                    print(e.stdout.decode(), flush=True)
                    stdout += e.stdout.decode()
                if e.stderr:
                    print(e.stderr.decode(), flush=True)
                    stderr += e.stderr.decode()
                timeout = None
            except:
                process.kill()
                raise
        retcode = process.poll()
        if retcode is None:
            raise RuntimeError('Process should have terminated')
        return subprocess.CompletedProcess(process.args, retcode,
                                           stdout, stderr)


def clear_processes(args):
    """Ensure all test processes have exited.

    Sometimes this doesn't happen, so we have to kill them.

    """
    exe = os.path.basename(args.test_ops)
    launcher_cmd = get_launcher_cmd(args.num_nodes, args, ppn=1)
    kill_cmd = ['killall', '-s', 'KILL', exe]
    # Not interested in the return.
    subprocess.run(launcher_cmd + kill_cmd, capture_output=True)


def run_test(args, num_procs, backend, operator, datatype, inplace,
             nonblocking, root):
    """Run a specified test."""
    launcher_cmd = get_launcher_cmd(num_procs, args)
    if args.extra_args:
        launcher_cmd += args.extra_args.split(' ')
    test_cmd = [args.test_ops,
                '--op', operator,
                '--backend', backend,
                '--datatype', datatype,
                # Keep things relatively small.
                '--max-size', '2048',
                # Don't wait too long.
                '--hang-timeout', str(hang_timeout),
                '--dump-on-error', '--max-dump-size', '64']
    if args.no_abort_on_hang:
        test_cmd.append('--no-abort-on-hang')
    if args.threads is not None:
        test_cmd += ['--threads', str(args.threads)]
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
    r = run_subprocess(launcher_cmd + test_cmd, timeout=(2*hang_timeout))
    if r.returncode == 0:
        print('[Pass] ' + test_desc, flush=True)
    else:
        print('[Fail] ' + test_desc, flush=True)
        if isinstance(r.args, str):
            print(r.args, flush=True)
        else:
            print(' '.join(r.args), flush=True)
        if r.stdout:
            print(r.stdout, flush=True)
        if r.stderr:
            print(r.stderr, flush=True)
    clear_processes(args)


def run_all_tests(args):
    """Run the full test suite."""
    procs = [2**x for x in range(
        int(math.log2(args.num_nodes * args.procs_per_node)) + 1)]
    if args.min_procs:
        procs = list(filter(lambda x: x >= args.min_procs, procs))
    for num_procs in procs:
        for backend, cases in test_cases.items():
            if args.backends and backend not in args.backends:
                continue
            for opdesc in cases['ops']:
                if args.operations and opdesc.op not in args.operations:
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
                            root_cases = [0] if opdesc.root else [None]
                            if num_procs > 1 and opdesc.root:
                                root_cases += [1]
                            for root in root_cases:
                                run_test(args, num_procs, backend, opdesc.op,
                                         datatype, inplace, nonblocking, root)


if __name__ == '__main__':
    run_all_tests(parser.parse_args())
