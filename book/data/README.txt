Stage inputs and outputs for the book pipeline.

Most of this directory is not tracked. The harvested PGNs, full label tables
and source Polyglot books come to roughly 1.3 GB and live outside the repo.
What is tracked is a small reference set: the positions the book must answer,
and a sample of our own engine labelling of them.

engines/   UCI engine binaries used for labelling. Not tracked, not present
           in the repository; create it and drop a binary in to run label/.
books/     Source Polyglot files used for gap-fill. Same.

See ../README.md for what each stage reads and writes.
