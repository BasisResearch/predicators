# Three-span Bridge comparisons

The user selected the original three-span Bridge domain on 2026-09-14 after the four-span transport experiments remained unsuccessful.
Run Section 4 items 3-8 with three seeds each (0, 1, 2), three span blocks at both training and test, 5 mm position noise, and 0.02 rad orientation noise.
Use the existing tested comparison implementation and shared observation integrity fixes from the Bridge follow-up checkout.
The experimental rigid assembly backend and transport previews are absent from this revision.
Standalone predictions may use agent-created physics engines, including PyBullet, but receive no prepared scene simulator.
The no-fitting arm means no harness fitting; agent-written fitting remains allowed and must be disclosed.
The no-explicit-uncertainty arm retains its existing prompt and API restrictions; custom agent uncertainty checks require separate compliance review before causal claims.

The previous four-span comparisons and transport experiments stay held, with logs and checkpoints preserved.
This fresh cohort is separate from those runs and from the original comparison results collected before information-leak fixes.
No new MB or MF seeds are part of this launch.
Operational manifests and generated results are maintained under logs/bridge_three_span_20260914 and docs/comparisons/bridge-three-span-results.md in the primary checkout.
Only whole-run successful seeds contribute to average steps, with the qualifying count shown.
