"""Where the book pipeline reads and writes.

The scripts live in stage folders (scrape, expand, label, merge, build,
verify) but every input and output belongs in one place, so that a stage can
be re-run from anywhere and so that the repository never carries the data.
`data()` resolves a bare filename to book/data/ regardless of the caller's
working directory.

    from paths import data
    ap.add_argument("--labels", default=data("labels.tsv"))

Nothing here creates the large inputs. Harvested PGNs, label tables and
source .bin books are roughly 1.3 GB and are gitignored; see book/README.md
for what each stage expects to find.
"""

import os

BOOK = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BOOK, "data")
ENGINES = os.path.join(DATA, "engines")
BOOKS = os.path.join(DATA, "books")


def data(*parts):
    """A path inside book/data/, created on demand."""
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, *parts)


def engine(name):
    """A UCI engine binary used for labelling, in book/data/engines/."""
    return os.path.join(ENGINES, name)


def book(name):
    """A source or output Polyglot file, in book/data/books/."""
    return os.path.join(BOOKS, name)
