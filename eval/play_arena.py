"""Human playback on ArenaEnv; trained-model evaluation remains in A3-012.

    python -m eval.play_arena --human --style direct --seed 0
    python -m eval.play_arena --human --style rotation --seed 0

The same application is available through arena.render, including its scripted demo.
No model is loaded and human play must be requested explicitly here.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from arena.play import main as play_main

    args = list(sys.argv[1:] if argv is None else argv)
    if '--human' not in args and not any(flag in args for flag in ('--help', '-h')):
        print('Trained-model evaluation is not implemented (A3-012). Use --human for keyboard play.',
              file=sys.stderr)
        return 2
    return play_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
