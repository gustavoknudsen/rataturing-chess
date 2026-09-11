"""Build the release executables from submission.zip.

The engine source comes out of submission.zip rather than out of src/, so the
released executable is provably the engine that played the tournament. The
only thing added is rataturing_uci.py, the protocol adapter.

Two variants are produced. They share every line of code and differ only in
whether net.npz is present, because that is what the engine itself keys its
evaluation on:

    NNUE     ships net.npz, plays with the trained network
    Classic  omits it, plays with the hand-crafted evaluation

The engine ships as ordinary .py files in engine/, never bundled into the
executable. numba finds its on-disk cache through each function's source path,
so the sources must stay real files at a stable location.

    python uci/build.py                    both variants
    python uci/build.py --variant nnue     one of them
    python uci/build.py --cache            enable numba's on-disk cache
    python uci/build.py --skip-smoke       skip the post-build check

--cache trades a slower first run for faster ones after it, roughly 85 s then
40 s against a flat 52 s. It is off by default so the shipped engine source is
byte for byte the submission.
"""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
SUBMISSION = os.path.join(REPO, "submission.zip")
ADAPTER = os.path.join(HERE, "rataturing_uci.py")
OUT = os.path.join(REPO, "release")
VERSION = "1.0"
VARIANTS = ("nnue", "classic")
# Bundled explicitly: PyInstaller analyses the adapter, and the adapter does
# not import these. Only the loose engine files do, and those are copied in
# after the freeze rather than analysed.
COLLECT = ("numba", "llvmlite", "numpy", "chess")


def _title(variant):
    return "NNUE" if variant == "nnue" else "Classic"


def _stage(variant, use_cache):
    """Unpack the submission into a per-variant engine folder."""
    staging = os.path.join(OUT, "staging", variant, "engine")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    with zipfile.ZipFile(SUBMISSION) as archive:
        archive.extractall(staging)
    if variant == "classic":
        net = os.path.join(staging, "net.npz")
        if os.path.exists(net):
            os.remove(net)
    if use_cache:
        _enable_cache(staging)
    return staging


def _enable_cache(staging):
    """Let numba persist compiled functions between runs. Seventeen of the
    engine's functions still refuse, because they read large module-level
    arrays that numba bakes in as pointers and cannot serialise, so this is
    a reduction in startup cost rather than an elimination of it."""
    changed = 0
    for name in sorted(os.listdir(staging)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(staging, name)
        text = io.open(path, encoding="utf-8").read()
        if "cache=False" not in text:
            continue
        changed += text.count("cache=False")
        io.open(path, "w", encoding="utf-8", newline="").write(
            text.replace("cache=False", "cache=True"))
    print("  cache enabled on %d functions" % changed)


def _freeze(variant):
    """Run PyInstaller over the adapter alone."""
    name = "Rataturing-" + _title(variant)
    work = os.path.join(OUT, "staging", variant)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
               "--console", "--name", name,
               "--distpath", os.path.join(work, "dist"),
               "--workpath", os.path.join(work, "work"),
               "--specpath", work]
    for package in COLLECT:
        command += ["--collect-all", package]
    command.append(ADAPTER)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit("PyInstaller failed for %s" % variant)
    return os.path.join(work, "dist", name), name


def _readme(folder, variant):
    text = """Rataturing %s %s
%s

A UCI chess engine. Written in Python, compiled at startup with numba.

This is the engine that played the AI Chessathon, unchanged, with a UCI
adapter around it. Source and full documentation:

    https://github.com/gustavoknudsen/Chess-Competition

FIRST, THE STARTUP TIME

The engine compiles itself when it starts, which takes about a minute. Your
GUI may look frozen during that time. It is not.

This happens once per session, not once per game. Leave the GUI open and
every game after the first starts instantly.

INSTALLING

Point your GUI at Rataturing-%s.exe. In Arena use Engines, Install New
Engine. In Cute Chess use Tools, Settings, Engines, Add.

Keep the folder intact. The exe needs engine/ beside it.

OPTIONS

    Hash            transposition table, MB. Default 256.
    OwnBook         play the opening book. Default true.
    Move Overhead   time reserved per move, ms. Default 50. Raise it if you
                    play over a network or lose on time.

The book covers the first 20 moves. It is skipped during analysis, so
infinite and fixed-depth searches always search.

WHAT IS IN THIS FOLDER

    Rataturing-%s.exe   the engine
    engine/             the competition engine, as ordinary Python files
    _internal/          Python runtime and libraries

EVALUATION

%s

LICENCE

MIT. See the repository.
""" % (_title(variant), VERSION, "=" * (12 + len(_title(variant))),
       _title(variant), _title(variant), _EVAL[variant])
    path = os.path.join(folder, "README.txt")
    io.open(path, "w", encoding="utf-8", newline="").write(text)


_EVAL = {
    "nnue": "Neural network, in engine/net.npz. King-bucketed, 32 buckets of\n"
            "768 features into a 512-wide layer, with 8 output buckets. This\n"
            "is the stronger build and the one that played the competition.",
    "classic": "Hand-crafted, with no network. Material, piece-square tables,\n"
               "pawn structure, king safety, mobility and threats. Weaker than\n"
               "the NNUE build. Included because it is a different engine to\n"
               "play against, and it needs no 25 MB network file.",
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _verify(folder, variant, use_cache):
    """Fail unless the engine about to ship is the submitted one.

    This is the whole reason the build reads submission.zip instead of src/.
    Without the check the claim would rest on the copy having been done right,
    which is exactly the kind of thing that quietly stops being true.
    """
    if use_cache:
        print("  engine not verified: --cache rewrites the njit decorators")
        return
    shipped = os.path.join(folder, "engine")
    checked = 0
    with zipfile.ZipFile(SUBMISSION) as archive:
        for info in archive.infolist():
            path = os.path.join(shipped, info.filename)
            if not os.path.exists(path):
                if variant == "classic" and info.filename == "net.npz":
                    continue
                raise SystemExit("missing from the build: " + info.filename)
            if _digest(archive.read(info.filename)) != \
                    _digest(io.open(path, "rb").read()):
                raise SystemExit("differs from submission.zip: "
                                 + info.filename)
            checked += 1
    print("  engine verified against submission.zip (%d files)" % checked)


def _smoke(exe):
    """Prove the built executable speaks UCI and returns a legal move."""
    print("  smoke test (allow a minute for the compile)")
    script = "uci\nisready\nposition startpos moves e2e4\ngo movetime 2000\nquit\n"
    try:
        result = subprocess.run([exe], input=script, capture_output=True,
                                text=True, timeout=400)
    except subprocess.TimeoutExpired:
        print("  FAIL timed out")
        return False
    out = result.stdout
    for expected in ("uciok", "readyok", "bestmove"):
        if expected not in out:
            print("  FAIL no %r in output" % expected)
            sys.stderr.write(out[-2000:] + result.stderr[-2000:])
            return False
    move = [l for l in out.splitlines() if l.startswith("bestmove")][0]
    print("  OK %s" % move)
    return True


def build(variant, use_cache, run_smoke):
    print("%s:" % _title(variant))
    staging = _stage(variant, use_cache)
    folder, name = _freeze(variant)
    target = os.path.join(folder, "engine")
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(staging, target)
    _readme(folder, variant)
    _verify(folder, variant, use_cache)
    exe = os.path.join(folder, name + ".exe")
    if run_smoke and not _smoke(exe):
        raise SystemExit("smoke test failed for %s" % variant)
    archive = os.path.join(OUT, "Rataturing-%s-%s-win64"
                           % (_title(variant), VERSION))
    if os.path.exists(archive + ".zip"):
        os.remove(archive + ".zip")
    shutil.make_archive(archive, "zip", os.path.dirname(folder),
                        os.path.basename(folder))
    size = os.path.getsize(archive + ".zip")
    print("  %s.zip (%.0f MB)" % (os.path.basename(archive), size / 1e6))
    return archive + ".zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variant", choices=VARIANTS, action="append",
                        help="build one variant; repeatable, default both")
    parser.add_argument("--cache", action="store_true",
                        help="enable numba's on-disk cache")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="do not run the built executable afterwards")
    args = parser.parse_args()
    if not os.path.exists(SUBMISSION):
        raise SystemExit("no submission.zip: run python tools/package.py first")
    os.makedirs(OUT, exist_ok=True)
    built = [build(v, args.cache, not args.skip_smoke)
             for v in (args.variant or list(VARIANTS))]
    print("\n%d archive(s) in %s" % (len(built), OUT))


if __name__ == "__main__":
    main()
