# Both policies lifted the target bowl. One completed the placement.

In LIBERO-Spatial task 0, the instruction identifies the black bowl between the plate and the ramekin. On initial state 12, both policies made two-sided contact with the requested bowl and lifted it. The released policy satisfied the placement predicate after 77 actions. The adapted policy ran for 280 actions and timed out. The shared seed was 1012, and the two saved simulator states immediately after settling had the same SHA-256 digest. [Released video](released.mp4) · [Adapted video](adapted.mp4) · [Frame-by-frame measurements](trace.csv)

![External and wrist cameras at reset, after action 68 and at termination. The successful and unsuccessful runs start from the same scene.](placement.jpg)

| Observation | Released checkpoint | Project-trained checkpoint |
| --- | ---: | ---: |
| First sampled two-pad contact on target | Frame 50 | Frame 48 |
| First target lift of at least 3 cm during contact | Frame 56 | Frame 52 |
| Greatest rise during contact | 9.73 cm | 7.48 cm |
| Bowl–plate horizontal separation at frame 68 | 0.76 cm | 5.70 cm |
| Bowl–plate horizontal separation at termination | 2.08 cm | 4.82 cm |
| Last sampled two-pad contact with target | Frame 77 | Frame 76 |

At frame 68 the released policy had carried the target over the plate center. The adapted policy had also transported the bowl toward the plate, but the centers were 5.70 cm apart and the bowl had already descended to 1.59 cm above its initial height. After frame 76, the target never again touched both fingerpad groups in a sampled frame. The final wrist view shows the bowl overlapping the plate in the image. LIBERO still marked the episode unsuccessful. Neither rollout registered two-sided contact with the other black bowl.

This pair illustrates a failure after a verified lift. Horizontal distance measures the separation of the simulated body positions in the table plane. LIBERO evaluates task success with its own predicate. Across all 100 adapted rollouts, 17 failures reached the lift threshold during target contact. [Failure counts and evaluation protocol](../README.md)
