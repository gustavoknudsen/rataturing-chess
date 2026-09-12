"""Build the staged submission: the layout that played the London final.

    python staged/build.py                 -> staged/staging/staged/
    python staged/build.py --zip           -> also writes submission_staged.zip

Built from submission.zip, like uci/build.py, so what ships is the engine that
competed with two files added and one renamed: agent.py becomes agent_real.py,
agent_staged.py takes its name, and agent_pure.py is the fallback. Every engine
file is hashed against the archive before the zip is written.
"""

import argparse
import hashlib
import io
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
SUBMISSION = os.path.join(REPO, "submission.zip")
OUT = os.path.join(HERE, "staging", "staged")
ZIP_OUT = os.path.join(REPO, "submission_staged.zip")

# 50 MB unzipped, the same cap package.py enforces.
MAX_UNZIPPED = 50_000_000


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    if not os.path.exists(SUBMISSION):
        raise SystemExit("no submission.zip: run python tools/package.py first")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    with zipfile.ZipFile(SUBMISSION) as archive:
        archive.extractall(OUT)
        names = archive.namelist()

    # The real engine gives up the name the platform imports.
    os.rename(os.path.join(OUT, "agent.py"), os.path.join(OUT, "agent_real.py"))
    shutil.copy(os.path.join(HERE, "agent_staged.py"),
                os.path.join(OUT, "agent.py"))
    shutil.copy(os.path.join(HERE, "agent_pure.py"),
                os.path.join(OUT, "agent_pure.py"))

    _verify(names)
    total = sum(os.path.getsize(os.path.join(OUT, f))
                for f in os.listdir(OUT))
    print("staged: %d files, %.2f MB unzipped (%.1f%% of cap)"
          % (len(os.listdir(OUT)), total / 1048576,
             100.0 * total / MAX_UNZIPPED))
    if total > MAX_UNZIPPED:
        raise SystemExit("over the 50 MB cap")
    return total


def _verify(names):
    """Every engine file must be byte-identical to the submission, and the
    rename must be the only difference."""
    with zipfile.ZipFile(SUBMISSION) as archive:
        for name in names:
            target = "agent_real.py" if name == "agent.py" else name
            path = os.path.join(OUT, target)
            if not os.path.exists(path):
                raise SystemExit("missing from the build: " + target)
            if _digest(archive.read(name)) != \
                    _digest(io.open(path, "rb").read()):
                raise SystemExit("differs from submission.zip: " + target)
    print("verified: %d engine files byte-identical to submission.zip"
          % len(names))


def write_zip():
    if os.path.exists(ZIP_OUT):
        os.remove(ZIP_OUT)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(os.listdir(OUT)):
            archive.write(os.path.join(OUT, name), arcname=name)
    print("wrote %s (%.2f MB zipped)"
          % (os.path.basename(ZIP_OUT), os.path.getsize(ZIP_OUT) / 1048576))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", action="store_true",
                        help="also write submission_staged.zip")
    args = parser.parse_args()
    build()
    if args.zip:
        write_zip()
    print("\nSmoke test it before trusting it:")
    print("  python tests/test_staged.py")


if __name__ == "__main__":
    main()
