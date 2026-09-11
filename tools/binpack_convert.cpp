// Convert a Stockfish .binpack training file into the flat arrays nnue_train.py
// reads, adding the one thing the Lichess evaluations database cannot give us:
// the game result, for WDL-blended training targets.
//
// Build (MSYS2 mingw64), from the repo root:
//   g++ -O2 -std=c++17 -I<nnue-pytorch>/data_loader/cpp/lib \
//       tools/binpack_convert.cpp -o tools/binpack_convert.exe
//
// Run:
//   tools/binpack_convert.exe in.binpack outdir [max_positions]
//
// **This is a local data-preparation tool and is never shipped.** The rules ban
// native binaries in the submission and third-party engines; a decoder for a
// training-data file format is neither, and training data is explicitly
// unrestricted. package.py has an explicit file list, so nothing here can leak
// into submission.zip.
//
// Conventions, which must match nnue_data.py exactly or the network trains on a
// board it will never see:
//   - square 0 = a8. This library uses square 0 = a1, hence `^ 56`.
//   - piece index = 6 * colour + type, type order P,N,B,R,Q,K. This library's
//     Color{White=0,Black=1} and PieceType{Pawn..King=0..5} already match.
//   - feature index = 64 * piece + square, padded to 32 slots with 65535.
//   - score is side-to-move relative, as it already is in this format.
//   - result is side-to-move relative too, in {-1,0,+1}; the reference loader
//     computes its target as (result + 1) / 2, so we store it unchanged and let
//     the trainer do that mapping.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>

// The upstream headers assume POSIX; mingw has gmtime_s with the arguments the
// other way round. Only the logging path in parallel_dataloader.h uses it.
#ifdef _WIN32
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

struct Writer {
    std::FILE* feats;
    std::FILE* counts;
    std::FILE* stm;
    std::FILE* score;
    std::FILE* result;
};

static std::FILE* open_out(const std::string& dir, const char* name) {
    std::string path = dir + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    return f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: binpack_convert in.binpack outdir [max]\n");
        return 1;
    }
    const std::string input = argv[1];
    const std::string outdir = argv[2];
    const std::uint64_t limit =
        (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0;

    Writer w{open_out(outdir, "feats.raw"), open_out(outdir, "counts.raw"),
             open_out(outdir, "stm.raw"), open_out(outdir, "score.raw"),
             open_out(outdir, "result.raw")};

    auto stream = training_data::open_sfen_input_file(input, false);
    if (!stream) {
        std::fprintf(stderr, "cannot open %s\n", input.c_str());
        return 1;
    }

    std::uint64_t seen = 0, kept = 0;
    std::uint64_t drop_eval = 0, drop_pieces = 0, drop_check = 0, drop_tac = 0;
    std::uint16_t row[MAX_PIECES];

    while (true) {
        auto entry = stream->next();
        if (!entry.has_value()) break;
        const TrainingDataEntry& e = *entry;
        ++seen;

        if (std::abs((int)e.score) > MAX_EVAL) { ++drop_eval; continue; }
        // A capture or promotion is about to invalidate the static evaluation,
        // which is what makes such positions bad training targets. Same filter
        // as nnue_data.py applies via its own best-move test.
        if (e.isCapturingMove() || e.move.promotedPiece != Piece::none()) {
            ++drop_tac;
            continue;
        }
        if (e.isInCheck()) { ++drop_check; continue; }

        int count = 0;
        bool overflow = false;
        for (int i = 0; i < MAX_PIECES; ++i) row[i] = PAD;
        for (Square sq : e.pos.piecesBB()) {
            if (count >= MAX_PIECES) { overflow = true; break; }
            Piece p = e.pos.pieceAt(sq);
            int piece = 6 * ordinal(p.color()) + ordinal(p.type());
            int square = ordinal(sq) ^ 56;
            row[count++] = (std::uint16_t)(64 * piece + square);
        }
        if (overflow || count < MIN_PIECES) { ++drop_pieces; continue; }

        std::uint8_t count8 = (std::uint8_t)count;
        std::uint8_t stm8 = (std::uint8_t)ordinal(e.pos.sideToMove());
        std::int16_t score16 = (std::int16_t)e.score;
        std::int8_t result8 = (std::int8_t)e.result;

        std::fwrite(row, sizeof(std::uint16_t), MAX_PIECES, w.feats);
        std::fwrite(&count8, 1, 1, w.counts);
        std::fwrite(&stm8, 1, 1, w.stm);
        std::fwrite(&score16, sizeof(std::int16_t), 1, w.score);
        std::fwrite(&result8, 1, 1, w.result);
        ++kept;

        if (kept % 5000000 == 0) {
            std::fprintf(stderr, "kept %llu of %llu seen\n",
                         (unsigned long long)kept, (unsigned long long)seen);
            std::fflush(stderr);
        }
        if (limit && kept >= limit) break;
    }

    std::fprintf(stderr, "done: kept %llu of %llu seen\n",
                 (unsigned long long)kept, (unsigned long long)seen);
    std::fprintf(stderr,
                 "dropped: eval %llu, pieces %llu, check %llu, tactical %llu\n",
                 (unsigned long long)drop_eval, (unsigned long long)drop_pieces,
                 (unsigned long long)drop_check, (unsigned long long)drop_tac);

    std::fclose(w.feats);
    std::fclose(w.counts);
    std::fclose(w.stm);
    std::fclose(w.score);
    std::fclose(w.result);
    return 0;
}
