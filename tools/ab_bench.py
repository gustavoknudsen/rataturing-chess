"""Interleaved A/B of the fixed-depth benchmark. Run:

    python ab_bench.py "A_LABEL:VAR=x,VAR2=y" "B_LABEL:VAR=z" [depth] [rounds]

Feature flags and tunables are frozen into the compiled code at import, so an
A/B has to be two processes. Wall time on this machine has been observed to
vary 60% between windows, which has produced wrong conclusions more than once,
so this alternates A,B,A,B... and reports the minimum of each. Load only ever
adds time, so the minimum is the honest estimate, and alternating means drift
hits both arms equally.

Nodes are printed alongside and are deterministic: if the two arms report the
same node count the change is a pure speed change, and only then is the time
comparison meaningful on its own.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_arm(spec):
    """"label:VAR=v,VAR2=v2" -> (label, {VAR: v, ...}). Env part may be empty."""
    label, _, assignments = spec.partition(":")
    env = {}
    for item in assignments.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        env[name.strip()] = value.strip()
    return label, env


def run_once(env_overrides, depth, history):
    env = dict(os.environ)
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "searchbench.py", str(depth), "1", str(history)],
        capture_output=True, text=True, cwd=ROOT, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-2000:])
    nodes = seconds = None
    for line in out.stdout.splitlines():
        if line.startswith("nodes "):
            nodes = int(line.split()[1])
        elif line.startswith("time "):
            seconds = float(line.split()[1].rstrip("s"))
    return nodes, seconds


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    arms = [parse_arm(sys.argv[1]), parse_arm(sys.argv[2])]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    history = int(sys.argv[5]) if len(sys.argv) > 5 else 60

    best = {label: (None, float("inf")) for label, _ in arms}
    for round_index in range(rounds):
        for label, env in arms:
            nodes, seconds = run_once(env, depth, history)
            if seconds < best[label][1]:
                best[label] = (nodes, seconds)
            print(f"  round {round_index + 1} {label:14s} "
                  f"{nodes:9d} nodes  {seconds:6.2f}s", flush=True)

    print()
    (label_a, _), (label_b, _) = arms
    nodes_a, time_a = best[label_a]
    nodes_b, time_b = best[label_b]
    print(f"{label_a:14s} {nodes_a:9d} nodes  {time_a:6.2f}s")
    print(f"{label_b:14s} {nodes_b:9d} nodes  {time_b:6.2f}s")

    node_delta = (nodes_b - nodes_a) / nodes_a * 100.0
    time_delta = (time_b - time_a) / time_a * 100.0
    print(f"\nB vs A: nodes {node_delta:+.1f}%  time {time_delta:+.1f}%")
    if nodes_a == nodes_b:
        print("identical node counts: pure speed change, time is the verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
