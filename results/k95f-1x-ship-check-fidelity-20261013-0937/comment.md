k95f result: the long-context half of the 1x v2.3.0 gate, on the other Spark, after a second fresh-user boot (new sparkrun HOME, empty runtime cache): up in 521 s, WARM.

Long-context retrieval (8k, 32k, 64k, 128k, plus 128k seeds 11 and 13, 20 questions each): PASS: 8000:20/20 32000:20/20 64000:20/20 128000:20/20 128000:20/20 128000:20/20.

```
depth    8000 | actual_tokens   15709 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   0.93s | lat    2.6s
depth   32000 | actual_tokens   61730 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   2.46s | lat    4.6s
depth   64000 | actual_tokens  122614 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   4.65s | lat    6.6s
depth  128000 | actual_tokens  245267 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   8.48s | lat   10.8s
depth  128000 | actual_tokens  244166 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   9.72s | lat   11.4s
depth  128000 | actual_tokens  244893 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft  10.21s | lat   12.6s
```

Results: {RESULTS_URL}
