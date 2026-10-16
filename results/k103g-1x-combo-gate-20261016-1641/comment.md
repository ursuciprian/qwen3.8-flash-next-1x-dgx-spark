k103g result: I ran the release quality gate of v2.3.0 on the k103 candidate (v2.3.0 plus the HC down projection BF16 path and GDN prefill views), one boot per Spark: tool use, stragglers and TC-45 on one, long-context retrieval on the other.

Gate: `PASS`.

```
k103g 2026-10-16 17:02:31 EEST: release quality gate on the k103 arm (ghcr.io/ursuciprian/spark-vllm-b12x:k103-20261015-a1ad1b69-10d0acc8, VLLM_GDN_PREFILL_VIEWS=1)
GATE=PASS
dgx01: views lines 1, plans measured 0, AOT load 2, Traceback 0
dgx02: views lines 1, plans measured 0, AOT load 2, Traceback 0
boot 151 s; pong pong
boot 145 s; pong pong
│    Quality:        92/100                                                    │
│    Score:  100 / 100                                                         │
│    Score:   100.0 ± 0.0 / 100                                                │
depth    8000 | actual_tokens   15709 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   0.89s | lat    2.7s
depth   32000 | actual_tokens   61730 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   2.32s | lat    4.5s
depth   64000 | actual_tokens  122614 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   4.35s | lat    6.3s
depth  128000 | actual_tokens  245267 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   8.01s | lat   10.4s
depth  128000 | actual_tokens  244166 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   9.20s | lat   11.2s
depth  128000 | actual_tokens  244893 | exact  20 near   0 wrong   0 explore   0 no_call   0 | typo_rate 0.00 | ttft   9.66s | lat   11.5s
c=7 round wall 6.3s | per request (submit order): #0:6.31s/320t #1:6.22s/320t #2:6.22s/320t #3:6.22s/320t #4:6.22s/320t #5:6.31s/320t #6:6.31s/320t
      preemptions +0 | queue time total 0.60s over 7 req | prefill total 3.25s | decode total 40.6s | accept 4.00/draft
c=8 round wall 6.5s | per request (submit order): #0:6.40s/320t #1:6.49s/320t #2:6.40s/320t #3:6.40s/320t #4:6.40s/320t #5:6.49s/320t #6:6.49s/320t #7:6.49s/320t
      preemptions +0 | queue time total 0.80s over 8 req | prefill total 3.80s | decode total 47.8s | accept 4.00/draft
c=12 round wall 11.3s | per request (submit order): #0:6.44s/320t #1:6.44s/320t #2:6.75s/320t #3:6.44s/320t #4:6.44s/320t #5:6.75s/320t #6:6.75s/320t #7:6.75s/320t #8:11.32s/320t #9:11.32s/320t #10:11.32s/320t #11:11.32s/320t
      preemptions +0 | queue time total 26.35s over 12 req | prefill total 30.84s | decode total 67.2s | accept 4.00/draft
c=16 round wall 12.9s | per request (submit order): #0:6.37s/320t #1:6.69s/320t #2:6.37s/320t #3:6.37s/320t #4:6.37s/320t #5:12.84s/320t #6:6.69s/320t #7:6.69s/320t #8:6.69s/320t #9:12.84s/320t #10:12.84s/320t #11:12.84s/320t #12:12.93s/320t #13:12.93s/320t #14:12.93s/320t #15:12.93s/320t
      preemptions +0 | queue time total 52.66s over 16 req | prefill total 59.11s | decode total 96.2s | accept 4.00/draft
```

Before a release the image still needs the warm build and a fresh-user boot check. Results: {RESULTS_URL}
