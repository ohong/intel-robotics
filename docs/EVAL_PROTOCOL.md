# Evaluation protocol — LEGO defect sorting

**Status:** partially specified; recording can proceed after the physical and labeling gates.
Scored autonomous evaluation waits on the missing task-specific fields below.
**Checked:** 2026-09-15 (PDT)

## 1. Authority and known rules

The supplied technical brief requires one closed loop: camera observation, Anomalib defect
detection, Physical AI Studio VLA reasoning, SO-101 action, and OpenVINO deployment on an
Intel Core Ultra Series 3 processor. The defect result must reach the robot workflow. The
brief says that the exact scenario, objects, operating conditions, and task constraints are
given at Challenge-Day kickoff. It does not define the LEGO defect or scoring rules.

The operator reports the current organizer task as:

- sort **LEGO blocks with defects into one pile**;
- sort **non-defective LEGO blocks into another pile**.

This report is recorded as an organizer requirement received from the operator. It is not
independently present in the supplied brief or in the public pages checked below.

## 2. Pending facts and recording gate

Record the authoritative handout, screen, or kickoff instruction that supplies each value.
Do not invent a value from an image, model output, or another team's submission.

| Field | Current value |
|---|---|
| Exact defect definition | **PENDING** |
| Acceptable normal variation | **PENDING** |
| Allowed LEGO block set and specimen identity | LEGO blocks reported; exact set **PENDING** |
| Required pile locations and acceptance condition | Two piles reported; geometry and final-state rule **PENDING** |
| Permitted initial placement, fixtures, workspace, and camera arrangement | **PENDING** |
| Allowed variation across presentations | **PENDING** |
| Operating conditions and safety limits | **PENDING** |
| Reinspection, reject, human-review, and retry rules | **PENDING** |
| Per-episode timeout and trial count | **PENDING** |
| Submission cutoff, timezone, and demo duration | **PENDING** |

Before the first physical recording, confirm only these gates:

1. The actual defect labels and acceptable normal variation, from the organizer or an
   authoritative inspection record.
2. The chosen layout, placement, fixtures, workspace, and camera arrangement are permitted.
3. One bounded supervised safety authorization covers the workspace limits, command owner,
   and reachable approved stop procedure.

Do not wait for the submission cutoff, demo duration, or final trial count before smoke
recording. Fill the timeout, recovery, final-state, and trial-count fields before scored
autonomous trials. Keep any provisional project default clearly labeled below.

## 3. Episode and split contract

An **episode** is one complete manual transfer or autonomous inspect-and-route attempt,
from the permitted initial state to a terminal outcome. One complete manual transfer is an
ACT episode even when detector and VLA inference do not run. The working unit is one LEGO
block per attempt, if the revealed layout permits that isolation. If the organizer requires
batch presentation, preserve each decision and its specimen identity while following the
organizer's episode boundary.

Every episode records the base fields:

`episode_id`, `episode_kind`, `skill_id`, `specimen_id`, `session_id`, `condition`, `route`,
`instruction`, robot state, action targets, timestamps, revision/configuration, and
intervention/failure fields. A manual ACT episode also records its task text and a sidecar
provenance manifest. It does not need detector or VLA fields. An autonomous trial adds the
synchronized observation ID, Anomalib output, VLA/action result, controller result, and
physical outcome evidence.

Split by complete episode and keep every group intact:

- **skill:** `normal-route` and `reject-route` are separate ACT skills. The normal route
  places a non-defective block in the non-defective pile. The reject route places a
  defective block in the defective pile. These are project training choices, not extra
  organizer rules.
- **specimen:** all views and episodes from one physical LEGO specimen stay in one split.
- **session:** all episodes from one contiguous capture session stay in one split. Plan
  separate train, validation/development, and final capture sessions intentionally; do not
  put all episodes in one session and then claim a session-level holdout.
- **frames:** never split adjacent frames from one episode across train, validation, and
  test. Near-duplicate derived images keep their source episode and specimen split.

Use these roles:

1. `train`: fit the route policy and training-only normalization statistics.
2. `validation/dev`: select a detector threshold, checkpoint, and repairs using held-out
   complete episodes. Do not use final trials for tuning.
3. `final`: fresh physical trials against the frozen system. Report the specimen/session
   overlap and limitation if fresh specimens are unavailable.

Record all attempted episodes, including failures, aborts, timeouts, and human rescues.
Do not report a successful streak without the failed attempts and counting rule.

### Smoke episodes

Record two short, complete **normal-route** smoke episodes before bulk collection, when
physical authorization and the revealed setup permit them. They establish recorder,
loader, timestamps, camera streams, state/action alignment, and replay compatibility.
If they are clean, have authoritative normal labels, and their specimen/session is assigned
to `train`, they may enter ACT policy training and the Anomalib normal training bank. Do not
blanket-discard them. They remain **excluded from scored autonomous results and final-trial
counts** and are not statistical performance evidence. The two episodes use the same route
by design; this does not validate the reject route.

## 4. Anomalib data separation

Use authoritative specimen labels. Never use the detector's own prediction as a ground-truth
label. Preserve image hash, source episode, specimen, session, camera, and capture time.

- **Anomalib train:** known-good normal LEGO specimens only. Fit the normal reference bank
  here, including an eligible normal smoke episode assigned to `train`. Defective images
  never enter this bank.
- **Anomalib validation:** normal and defective images from held-out specimens/sessions.
  Select the threshold here, with labels supplied by an authoritative inspection record.
- **Anomalib test:** fresh held-out normal and defective specimens/sessions. Freeze the
  model and threshold before this measurement.

The policy and anomaly splits may be derived from the same finalized episode only when
their specimen/session group remains intact. Deduplicate exact bytes and flag near-duplicate
frames. A long video of one block is not independent specimen evidence.

## 5. Outcome and failure record

Project pilot defaults below are recommendations, not organizer rules. Update them when
the exact kickoff handout arrives:

- **Success:** after release, the block is entirely in the chosen correct pile, remains
  there for two seconds, and the gripper clears it.
- **Timeout:** 60 seconds maximum per episode.
- **Recovery:** no retry or human rescue counts as success. Record either as an intervention
  or failure; use a retry only if the organizer permits it.
- **Fresh pilot matrix:** two conditions (normal/non-defective and defective) × three
  permitted pickup positions × two permitted orientations = 12 setups. Reduce the matrix
  honestly if the task disallows a position or orientation, and do not fabricate variation.

The organizer's pile acceptance, retention, clearance, timeout, retry, and trial-count rules
remain pending. Do not present these project defaults as official requirements.

For each attempt record:

- correct/incorrect defect decision and route;
- correct-pile placement and physical outcome evidence;
- grasp, drop, contact, timeout, stale-input, invalid-action, and hardware/software faults;
- intervention or rescue, retry/reinspection, and whether the organizer permits it;
- detector latency, VLA/action latency, observation-to-command delay, and full cycle time.

Count human intervention as an intervention in final results. Treat an ambiguous detector
result as hold/review/stop unless the revealed rules authorize reinspection or recovery.

Label every result as `REAL`, `RECORDED_REAL`, `REPLAY`, `SYNTHETIC`, or `MOCK`.
Two smoke episodes, offline action loss, a simulator, and detector-only measurements do
not prove autonomous physical task success.

## 6. Verified sources and limits

- [Supplied Technical Brief](Physical%20AI%20Challenge%20Technical%20Brief.pdf), pages 1–4.
  The two repository copies are byte-identical: SHA-256
  `c70b8325d99b705cccbdf736df54805b8a840cef8fbee006a2fab32d1cfa81a1`.
- [Intel Newsroom — Intel at AI Infra Summit 2026](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-at-ai-infra-summit-2026.html), published 2026-08-26,
  checked 2026-09-15. It lists the virtual challenge as September 10–16, the onsite
  challenge as September 15–16, and final submissions/judging as September 16. It gives
  no cutoff time, timezone, demo duration, or LEGO task rule.
- [AI Infra Summit Hackathon](https://www.ai-infra-summit.com/ai-infra-hackathon), checked
  2026-09-15. It describes a two-day September 15–16 hackathon and says full sponsor-track
  details are published ahead of the event. The page gives no exact task fields or cutoff
  time. The public page therefore does not resolve the pending values above.

The next required input is the organizer's exact kickoff rule set: defect definition and
normal variation, permitted block presentation and workspace, pile acceptance, timeout /
trial count / recovery policy, and the submission cutoff with timezone and demo duration.
