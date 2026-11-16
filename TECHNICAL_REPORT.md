# NALAR technical report

Team NALAR, More than a Hack 2027 Phase 1. Source repository:
https://github.com/tearsbear/nalar-more-than-a-hack-2027

## Method and inputs

NALAR uses a deterministic state machine for home possession (H), away
possession (A), dead ball (D) and unknown (X). It reads the supplied tactical
tracking metadata, raw tracking and automatic Smart Tagging. It uses neither
video, broadcast tracking nor homography. Tactical tracking already provides
pitch positions on the submission timeline, so the pipeline needs no video
processing or alignment to the broadcast.

The parser requires tracking frame IDs 1 to 199,319 in order and subtracts one
for the submission index. It scales normalized positions to the 105.3 × 68 m
pitch and computes the observed ball position and the distance to each team's
nearest valid player. Player slots are independent observations, with no
assumption of stable identities. Off-pitch positions are retained for exit
detection.

There is no training dataset or fitted model. The provided game's human labels
and the organizer's public GT do not inform the inference rules or parameter
choices. We used OpenAI Codex to assist with development, review and
documentation. The pipeline runs locally and uses no LLM or external service
to generate labels.

## Possession and restarts

During live play, the previous team keeps possession through loose balls,
brief touches and missing observations until another team establishes control.
A geometry vote requires a player within 2.5 m of the ball and at least 1 m
closer than the other team's nearest player. Control requires at least 10 votes
in 25 frames, at least 80% agreement among valid votes, and support from the
latest vote. A takeover starts at the first supporting vote after the last
opposing vote in that window. Confirmation history resets at stoppages and
half boundaries.

Build-up, progression, chance creation and attacking-transition tags can
support a change after 50 consecutive missing-ball frames with an unambiguous
tag. Opposing tags cancel, and confirmed geometry takes precedence. Short-gap
interpolation is diagnostic only. It supplies no votes or restart evidence.

A ball observed more than 1 m beyond the pitch for 10 consecutive frames
establishes an exit from the first frame of the run. Within a set-piece tag,
50 consecutive observed frames with a ball-position bounding-box diagonal of
at most 0.75 m establish a waiting period. A tagged restart uses five
consecutive observed motion samples above 2 m/s in bounds. The restart is
backdated to the onset of qualifying motion. A confirmed restart or published
kickoff ends waiting periods already in progress, including overlapping
ones. A later waiting period can establish a new stoppage. Confirmed exits
take precedence.

A restart can inherit an unambiguous set-piece team. With an unknown team,
NALAR fills the episode back to the restart once it confirms control. If the
ball exits first, the unresolved prefix stays X. Outside tagged restarts, the
dead-ball state can end after control confirms during the current sustained
motion run or after sustained attacking tags during a tracking gap. A stop
or missing observation invalidates prior motion evidence.

The published half ranges and kickoff metadata define the frame mask. NALAR
writes X outside the two graded ranges and D during the published second-half
wait before kickoff. The submission has 158,857 graded frames and 40,462
ungraded frames. Of the graded frames, 111 remain X after unresolved restarts
end in exits. On graded frames, macroF1 counts X as a miss. SpellF1 fills X
within scored runs according to the [official evaluation rules](EVALUATION.md).

## Verification and reproduction

All 66 tests passed, including synthetic regressions for possession changes,
restart ordering and output format. Full-match generation from an isolated
source tree with the pinned dependencies reproduced the exact CSV. The file
passed the official format validator. These checks do not measure full-match
accuracy, and the release candidate was not scored against public GT.

Follow the setup and hash check in [REPRODUCTION.md](REPRODUCTION.md). From the
repository root, with the environment activated and `MTAH_DATA_DIR` set to the
separately supplied bundle, run:

```bash
python -m nalar generate --data-dir "$MTAH_DATA_DIR" \
  --config configs/nalar_v1.json --output artifacts/submission/NALAR.csv \
  --manifest artifacts/submission/run.json
python evaluation/validate_submission.py artifacts/submission/NALAR.csv
```

## Limitations

Distance to the ball is an imperfect proxy for control, especially on
deflections, aerial balls and noisy tracks. Long occlusions can hide turnovers
or stoppages. Coarse phase tags can delay or misplace a change, and the
stationarity and motion rules can confuse repositioning with a restart or miss
a stoppage when the ball is unobserved. The method needs independent validation
on other matches to measure these errors.
