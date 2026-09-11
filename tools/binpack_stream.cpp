// Stream a Stockfish .binpack straight into training batches, with no
// intermediate files at all.
//
// Build (Linux, from the repo root):
//   g++ -O2 -std=c++20 -I<nnue-pytorch>/data_loader/cpp/lib \
//       tools/binpack_stream.cpp -o tools/binpack_stream
//
// Run:
//   tools/binpack_stream a.binpack b.binpack ... > pipe
//
// **Why this exists alongside binpack_convert.** The batch converter
// materialises the corpus: 2B positions become 128 GB of .raw, then a two-pass
// external shuffle writes another 128 GB. That is fine on a machine with a
// spare terabyte and five hours, and impossible anywhere else - a Kaggle
// session has about 70 GB of disk in total. Streaming removes the corpus size
// from the equation entirely: the only thing on disk is the binpack itself,
// which holds 2B positions in 10 GB. It also removes the shuffle pass, because
// the reader shuffles in RAM.
//
// The filters, conventions and record fields are deliberately identical to
// binpack_convert.cpp so that a streamed net and a materialised net are
// trained on the same distribution and remain comparable. The book filter,
// which lives in binpack_prep.py for the materialised path, has to move here
// because there is no later pass to apply it.
//
// This is local tooling and is never shipped; package.py has an explicit file
// list. The rules ban native binaries in the submission, not on the machine
// that prepares training data.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static inline struct tm* gmtime_r(const time_t* clock, struct tm* result) {
    return gmtime_s(result, clock) == 0 ? result : nullptr;
}
#endif

#include "nnue_training_data_stream.h"

using namespace binpack;
using namespace chess;

static const int MAX_PIECES = 32;
static const std::uint16_t PAD = 65535;
static const int MAX_EVAL = 10000;
static const int MIN_PIECES = 4;

// One position, packed to 69 bytes with no padding, matching the numpy
// structured dtype in nnue_stream.py field for field.
#pragma pack(push, 1)
struct Record {
    std::uint16_t feats[MAX_PIECES];
    std::uint8_t count;
    std::uint8_t stm;
    std::int16_t score;
    std::int8_t result;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 69, "record must be packed to 69 bytes");

struct Counters {
    std::uint64_t seen = 0, kept = 0;
    std::uint64_t eval = 0, pieces = 0, check = 0, tactical = 0, book = 0;
};

// Both kings and all four rooks still at home with full material: the
// signature of a position still in the opening book. Leela self-play games all
// start from the same place, so each game donates its opening and the corpus
// ends up 1.35% literal start positions. Square 0 = a8, feature = 64 * piece +
// square, so e1 = 60, a1 = 56, h1 = 63, e8 = 4, a8 = 0, h8 = 7.
static bool is_book(const std::uint16_t* row, int count) {
    if (count != 32) return false;
    bool wk = false, wra = false, wrh = false;
    bool bk = false, bra = false, brh = false;
    for (int i = 0; i < count; ++i) {
        std::uint16_t f = row[i];
        if (f == 5 * 64 + 60) wk = true;
        else if (f == 3 * 64 + 56) wra = true;
        else if (f == 3 * 64 + 63) wrh = true;
        else if (f == 11 * 64 + 4) bk = true;
        else if (f == 9 * 64 + 0) bra = true;
        else if (f == 9 * 64 + 7) brh = true;
    }
    return wk && wra && wrh && bk && bra && brh;
}

// Fills `out` from one entry, or returns false if the entry is filtered out.
static bool encode(const TrainingDataEntry& e, Record& out, Counters& c) {
    if (std::abs((int)e.score) > MAX_EVAL) { ++c.eval; return false; }
    // A capture or promotion is about to invalidate the static evaluation,
    // which is what makes such positions bad training targets.
    if (e.isCapturingMove() || e.move.promotedPiece != Piece::none()) {
        ++c.tactical;
        return false;
    }
    if (e.isInCheck()) { ++c.check; return false; }

    int count = 0;
    for (int i = 0; i < MAX_PIECES; ++i) out.feats[i] = PAD;
    for (Square sq : e.pos.piecesBB()) {
        if (count >= MAX_PIECES) { ++c.pieces; return false; }
        Piece p = e.pos.pieceAt(sq);
        int piece = 6 * ordinal(p.color()) + ordinal(p.type());
        out.feats[count++] = (std::uint16_t)(64 * piece + (ordinal(sq) ^ 56));
    }
    if (count < MIN_PIECES) { ++c.pieces; return false; }
    if (is_book(out.feats, count)) { ++c.book; return false; }

    out.count = (std::uint8_t)count;
    out.stm = (std::uint8_t)ordinal(e.pos.sideToMove());
    out.score = (std::int16_t)e.score;
    out.result = (std::int8_t)e.result;
    return true;
}

static void run_file(const std::string& path, Counters& c,
                     std::vector<Record>& buf) {
    auto stream = training_data::open_sfen_input_file(path, false);
    if (!stream) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    while (true) {
        auto entry = stream->next();
        if (!entry.has_value()) break;
        ++c.seen;
        Record rec;
        if (!encode(*entry, rec, c)) continue;
        buf.push_back(rec);
        ++c.kept;
        if (buf.size() == buf.capacity()) {
            std::fwrite(buf.data(), sizeof(Record), buf.size(), stdout);
            buf.clear();
        }
        if (c.kept % 20000000 == 0) {
            std::fprintf(stderr, "streamed %llu of %llu seen\n",
                         (unsigned long long)c.kept,
                         (unsigned long long)c.seen);
            std::fflush(stderr);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: binpack_stream in.binpack [more...]\n");
        return 1;
    }
#ifdef _WIN32
    // stdout must not translate 0x0A into 0x0D 0x0A or every record shifts.
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    Counters c;
    std::vector<Record> buf;
    buf.reserve(16384);
    for (int i = 1; i < argc; ++i) run_file(argv[i], c, buf);
    if (!buf.empty())
        std::fwrite(buf.data(), sizeof(Record), buf.size(), stdout);
    std::fflush(stdout);
    std::fprintf(stderr,
                 "done: kept %llu of %llu seen\n"
                 "dropped: eval %llu, pieces %llu, check %llu, "
                 "tactical %llu, book %llu\n",
                 (unsigned long long)c.kept, (unsigned long long)c.seen,
                 (unsigned long long)c.eval, (unsigned long long)c.pieces,
                 (unsigned long long)c.check, (unsigned long long)c.tactical,
                 (unsigned long long)c.book);
    return 0;
}
