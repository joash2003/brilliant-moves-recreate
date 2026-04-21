"""
scrape_lichess.py -- fetch public Lichess studies and extract PGNs.

Reproduces the data-acquisition half of Stage 2 of the paper. The paper uses
624 Lichess Studies authored by a specific user known to annotate "brilliant"
moves. Those IDs were not released, so this utility is best used with either:

  (1) a list of known study IDs (one per line in --ids_file)
  (2) a range of studies authored by a single user (--user)

Lichess exposes a public REST API that returns the full study as PGN. No
auth token is required for public studies; an optional token can be supplied
via --token to raise the rate limit.

Workflow:
  python scrape_lichess.py --ids_file studies.txt --pgn_out raw_pgns/
  python pgn_parser.py raw_pgns/merged.pgn          # produces moves/
  # then hand-label: write moves/<name>/label.txt as "brilliant" / "not_brilliant"

References:
  https://lichess.org/api#tag/Studies/operation/studyAllChaptersPgn
"""

import argparse
import os
import time

import requests


LICHESS_STUDY_PGN = "https://lichess.org/api/study/{sid}.pgn"
LICHESS_USER_STUDIES = "https://lichess.org/api/study/by/{user}/export.pgn"


def fetch_study(sid: str, token: str | None = None, timeout: float = 30.0) -> str:
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(LICHESS_STUDY_PGN.format(sid=sid), headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_by_user(user: str, token: str | None = None, timeout: float = 60.0) -> str:
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(LICHESS_USER_STUDIES.format(user=user), headers=headers, timeout=timeout, stream=True)
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_file", help="Text file with one Lichess study ID per line.")
    ap.add_argument("--user", help="Download all public studies from a Lichess user.")
    ap.add_argument("--pgn_out", default="raw_pgns", help="Directory to write PGN files into.")
    ap.add_argument("--merge", action="store_true", help="Also emit a single merged.pgn in pgn_out.")
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"), help="Optional Lichess API token (env: LICHESS_TOKEN).")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests (be kind to lichess).")
    args = ap.parse_args()

    os.makedirs(args.pgn_out, exist_ok=True)
    merged_parts: list[str] = []

    if args.user:
        print(f"[scrape] fetching all studies for user {args.user!r}")
        pgn = fetch_by_user(args.user, token=args.token)
        out = os.path.join(args.pgn_out, f"{args.user}.pgn")
        with open(out, "w", encoding="utf-8") as f:
            f.write(pgn)
        merged_parts.append(pgn)
        print(f"[scrape] wrote {out} ({len(pgn):,} chars)")

    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        print(f"[scrape] fetching {len(ids)} studies")
        for i, sid in enumerate(ids, 1):
            try:
                pgn = fetch_study(sid, token=args.token)
            except Exception as e:
                print(f"[scrape]   ({i}/{len(ids)}) {sid}: FAILED ({e})")
                continue
            out = os.path.join(args.pgn_out, f"{sid}.pgn")
            with open(out, "w", encoding="utf-8") as f:
                f.write(pgn)
            merged_parts.append(pgn)
            print(f"[scrape]   ({i}/{len(ids)}) {sid}: {len(pgn):,} chars")
            time.sleep(args.sleep)

    if args.merge and merged_parts:
        merged_path = os.path.join(args.pgn_out, "merged.pgn")
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(merged_parts))
        print(f"[scrape] merged -> {merged_path}")


if __name__ == "__main__":
    main()
