# Draft email to Kamron Zaidi — DO NOT SEND UNTIL REVIEWED

**To:** kamron.zaidi@mail.utoronto.ca
**Cc (optional):** guerzhoy@cs.toronto.edu
**Subject:** Request for training dataset — "Predicting User Perception of Move Brilliance in Chess" (arXiv 2406.11895)

---

Dear Kamron,

I hope this message finds you well. I am a [YOUR PROGRAM / YEAR, e.g. "final-year BSc Informatics"] student at [YOUR UNIVERSITY], and my practical/thesis work is focused on reproducing your paper "Predicting User Perception of Move Brilliance in Chess" (ICCC 2024, arXiv:2406.11895). Thank you for releasing the `brilliant-moves-clf` repository — the `search.cc` patch, pretrained AggReduce model, and demo game trees have been extremely helpful, and I have successfully reproduced the inference pipeline on the two demo positions (Byrne–Fischer 1956 and Vranesic–Stein) on my own machine.

To carry out the full reproduction — including training from scratch and comparing against the numbers in Table 1 — I would very much appreciate access to the training data used in the paper. Specifically:

1. The list of 624 Lichess Study identifiers collected on 13 November 2023 (or the raw PGNs/JSON), and
2. The per-move annotation labels you derived from them (the 820 brilliant / 1637 good / 4518 other splits), ideally keyed by game + ply so I can align them with independently scraped PGNs.

If that dataset is too large or encumbered to share directly, even a list of the 624 study IDs plus the train/val/test split indices would be enormously valuable, since the rest can be re-derived from the Lichess API.

I fully understand if this is not possible — in parallel I am independently scraping the most-popular Lichess Studies via the public API and documenting any dataset differences, per the methodology in Section "Data Collection" of the paper. Any guidance you can share about edge cases (e.g. how multi-variation studies were handled, or how duplicate positions across studies were deduplicated) would still be very welcome.

Please let me know if you have any questions, and thank you again for the thoughtful and reproducible release.

Best regards,
[YOUR FULL NAME]
[YOUR UNIVERSITY AND PROGRAM]
[YOUR EMAIL]

---

## Notes for you before sending

- Replace the bracketed placeholders with your real details.
- Keep it short and specific — researchers respond to concrete asks more readily than vague ones.
- Cc'ing Prof. Guerzhoy is optional but increases reply odds and is professional courtesy.
- Reasonable wait: 5–7 working days before following up once.
