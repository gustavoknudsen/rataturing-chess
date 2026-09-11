"""Non-atomic numba reference counting. Import this before anything compiles.

Numba emits an atomic incref/decref pair for every array argument of every
njit -> njit call. The engine passes 22 arrays per node, and on x86 those
`lock`-prefixed updates are a real cost. Single-threaded code does not need
the atomicity, so this replaces the pair with plain loads and stores.

Numba's own `_disable_atomicity` is broken in 0.67.0 - its non-atomic path
returns the old value where `NRT_decref` expects the new one, so nothing is
ever freed. This supplies a corrected replacement.

**Only correct because the agent is single-threaded.** No prange, no nogil,
no threads. Do not carry this into anything parallel.

Applied after the first compile it silently does nothing, so import it first.
"""

import os

# Defensive: this module is an optimisation, never a requirement. llvmlite is a
# hard dependency of numba so it is always present in practice, but if any of
# this ever fails to import - a numba version that moves nrtdynmod, say - the
# engine must still start. Losing the speedup is survivable; failing at import
# loses every game.
try:
    from llvmlite import ir
    from numba.core.runtime import nrtdynmod
    _AVAILABLE = True
except Exception:                                    # pragma: no cover
    _AVAILABLE = False


def _nonatomic_inc_dec(module, op, ordering):
    """Drop-in for nrtdynmod._define_atomic_inc_dec, minus the lock prefix.

    Returns the NEW value, which is what NRT_decref tests against zero to
    decide whether to call the destructor. Numba's own non-atomic branch
    returns the old value and therefore leaks every allocation.
    """
    ftype = ir.FunctionType(nrtdynmod._word_type,
                            [nrtdynmod._word_type.as_pointer()])
    fn = ir.Function(module, ftype, name=f"nrt_atomic_{op}")
    [ptr] = fn.args
    builder = ir.IRBuilder(fn.append_basic_block())
    one = ir.Constant(nrtdynmod._word_type, 1)
    old = builder.load(ptr)
    new = getattr(builder, op)(old, one)
    builder.store(new, ptr)
    builder.ret(new)
    return fn


def apply():
    """Idempotent. Returns True if the patch is in place."""
    if not _AVAILABLE:
        return False
    try:
        nrtdynmod._define_atomic_inc_dec = _nonatomic_inc_dec
    except Exception:                                # pragma: no cover
        return False
    return nrtdynmod._define_atomic_inc_dec is _nonatomic_inc_dec


# BTC_NRT=0 leaves numba's atomic refcounts alone, so the change can be A/B'd
# in a single directory. On by default.
_APPLIED = apply() if os.environ.get("BTC_NRT", "1") == "1" else False
