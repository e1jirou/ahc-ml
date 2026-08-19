from __future__ import annotations

import argparse
import sys

from .simulation import generate_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one prerecorded AHC015 case")
    parser.add_argument("--seed", type=int, default=15015)
    args = parser.parse_args()
    flavors, ranks = generate_cases(1, args.seed)
    print(" ".join(map(str, flavors[0])), file=sys.stdout)
    for rank in ranks[0]:
        print(int(rank), file=sys.stdout)


if __name__ == "__main__":
    main()
