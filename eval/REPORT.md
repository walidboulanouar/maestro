# Maestro eval report

Reproduce: `npm run eval`. Offline + deterministic: routes over the priced
registry but executes on the mock provider, graded against each fixture's
ground-truth difficulty with oracle-route regret. 25 fixtures, 7 models.

| strategy | pass% | mean $ | pass/$ | regret $ | fails |
|---|--:|--:|--:|--:|--:|
| maestro | 92% | 0.00053 | 1747.9 | 0.00035 | 2 |
| best-single | 100% | 0.01507 | 66.3 | 0.01421 | 0 |
| cheapest-single | 56% | 0.00016 | 3566.0 | 0.00000 | 11 |
| rule-only | 76% | 0.00036 | 2130.6 | 0.00011 | 6 |
| random-route | 88% | 0.00689 | 127.7 | 0.00705 | 3 |

calibration: Brier = 0.181, ECE = 0.110 (lower is better).

**Read:** maestro reaches 92% of best-single quality at 97% lower mean cost, and beats rule-only and random on quality. cheapest-single is cheaper per call but fails 11/25. Not a leaderboard claim.
