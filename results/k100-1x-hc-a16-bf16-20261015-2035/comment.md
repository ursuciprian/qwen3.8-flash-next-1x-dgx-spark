k100-1x-hc-a16-bf16 result: a BF16 path for the hyperconnection down projection (K10240 to N336) in prefill, against the 1x v2.3.0 stack.

Background: k96 showed that the down projection is served A16 (BF16 activations, plan record a16 n128 k64 s1, 1.883 ms at M 6048), so k90's quantized 128x128/BK64 candidate does not apply to the served path. The A16 kernel only has 16-row M tiles and dequantizes the 3.4 MB weight again for every 16 rows. My b12x branch exp/k100-hc-a16-bf16 (5b77e18e) sends prefill calls of 1024 rows or more to an exact BF16 dequantization of the weight plus one BF16 GEMM with FP32 accumulation. Decode (exact-M regimes), smaller calls and every other shape keep the A16 kernel. Plan records and plan population do not change.

The job runs a probe first: identical captured inputs and real layer 5 weights, the served plan seed as control, numerics before speed (k96 gate), decode rows M 1 to 40, the HC norm/mix kernels as separate rows, and the TP=2 row on the 2x image. The serving A/B (4 boots per side per Spark, ctl arm arm ctl ctl arm arm ctl, cold TTFT 16K/64K/128K, llama-benchy c1/c4/c8, MTP acceptance per position) only runs when the probe projects at least 2% on cold 16K TTFT.

RULE=GO (probe PREDICT=GO deq-pc +0.296 s (+3.8%), arm XCHECK=PASS; serving A/B next: k100ab (both Sparks); 2x row PREDICT=GO deq-pc +0.260 s (+4.7%))

```
k100 2026-10-15 20:40:22 EEST: b12x A16 BF16 large-row path for the HC down projection (arm image ghcr.io/ursuciprian/spark-vllm-b12x:k100-20261015-a1ad1b69-5dad364d, b12x a1ad1b69) vs 1x v2.3.0
RULE=GO (probe PREDICT=GO deq-pc +0.296 s (+3.8%), arm XCHECK=PASS; serving A/B next: k100ab (both Sparks); 2x row PREDICT=GO deq-pc +0.260 s (+4.7%))

== P0 probe, 1x v2.3.0 image (served plan seed 33d9b32e)
setup 1x: seed 33d9b32e, 16K chunks [6048, 6048, 3024, 1264], cold 16K TTFT 7.71 s (k90 arm dgx-01, 2125 tok/s)
inputs: captured 16384 rows; weights: layer 5 attention HC, online MXFP8 g32

== down K10240 N336: full-op ms per call, x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails. Control: ctl-seed at capacity 8192, ctl (forced served config) above
cap      M           ctl-seed            deq-pc               ctl        deq-cached           deq-b16   control plan
 8192    256              0.072      0.160 x0.45~               n/a      0.050 x1.45~      0.155 x0.47~   a16 n128 k64 s1 [seed hit] -
 8192    512              0.136      0.195 x0.70~               n/a      0.074 x1.83~      0.180 x0.75~   a16 n128 k64 s1 [seed hit] -
 8192   1024              0.267      0.273 x0.98~               n/a      0.155 x1.73~      0.259 x1.03~   a16 n128 k64 s1 [seed hit] -
 8192   1264              0.337      0.284 x1.18=               n/a      0.173 x1.95=      0.276 x1.22=   a16 n128 k64 s1 [seed hit] -
 8192   2048              0.674      0.374 x1.80=               n/a      0.272 x2.47=      0.360 x1.87=   a16 n128 k64 s1 [seed hit] -
 8192   2064              0.686      0.383 x1.79=               n/a      0.286 x2.40=      0.375 x1.83=   a16 n128 k64 s1 [seed hit] -
 8192   2864              0.925      0.474 x1.95=               n/a      0.358 x2.58=      0.457 x2.03=   a16 n128 k64 s1 [seed hit] -
 8192   3024              0.946      0.475 x1.99=               n/a      0.375 x2.52=      0.461 x2.05=   a16 n128 k64 s1 [seed hit] -
 8192   4536              1.391      0.700 x1.99=               n/a      0.594 x2.34=      0.662 x2.10=   a16 n128 k64 s1 [seed hit] -
 8192   5728              1.850      0.829 x2.23=               n/a      0.731 x2.53=      0.810 x2.28=   a16 n128 k64 s1 [seed hit] -
 8192   6048              1.921      0.859 x2.24=               n/a      0.755 x2.55=      0.803 x2.39=   a16 n128 k64 s1 [seed hit] -
 8192   8192              2.563      1.176 x2.18=               n/a      1.055 x2.43=      1.070 x2.39=   a16 n128 k64 s1 [seed hit] -
 9072   9072                n/a      1.273 x2.23=             2.838               n/a               n/a   a16 n128 k64 s1 -
12096  12096                n/a      1.651 x2.24=             3.699               n/a               n/a   a16 n128 k64 s1 -

== up K320 N10240: full-op ms per call, x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails. Control: ctl-seed at capacity 8192, ctl (forced served config) above
cap      M           ctl-seed               ctl            deq-up   control plan
 8192    256              0.076               n/a      0.231 x0.33!   quantized [seed hit] (64, 128)/BK128/u1
 8192    512              0.078               n/a      0.354 x0.22!   quantized [seed hit] (64, 128)/BK128/u1
 8192   1024              0.140               n/a      0.610 x0.23!   quantized [seed hit] (64, 128)/BK128/u1
 8192   1264              0.171               n/a      0.772 x0.22!   quantized [seed hit] (64, 128)/BK128/u1
 8192   2048              0.273               n/a      1.158 x0.24!   quantized [seed hit] (64, 128)/BK128/u1
 8192   2064              0.273               n/a      1.218 x0.22!   quantized [seed hit] (64, 128)/BK128/u1
 8192   2864              0.354               n/a      1.574 x0.22!   quantized [seed hit] (64, 128)/BK128/u1
 8192   3024              0.391               n/a      1.641 x0.24!   quantized [seed hit] (64, 128)/BK128/u1
 8192   4536              0.539               n/a      2.557 x0.21!   quantized [seed hit] (64, 128)/BK128/u1
 8192   5728              0.668               n/a      2.970 x0.22!   quantized [seed hit] (64, 128)/BK128/u1
 8192   6048              0.723               n/a      3.366 x0.21!   quantized [seed hit] (64, 128)/BK128/u1
 8192   8192              0.938               n/a      4.436 x0.21!   quantized [seed hit] (64, 128)/BK128/u1
 9072   9072                n/a             1.034               n/a   quantized (64, 128)/BK128/u1
12096  12096                n/a             1.460               n/a   quantized (64, 128)/BK128/u1

== decode/drafter rows (capacity 8192, exact-M regimes [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35, 40]): ms per call, plan; the job compares their bytes between the shipped and the arm image
  down: M1 0.0650 (a16 n64 k128 s4, exact)  M2 0.0639 (a16 n64 k128 s8, exact)  M3 0.0639 (a16 n64 k128 s8, exact)  M4 0.0636 (a16 n64 k128 s4, exact)  M5 0.0640 (a16 n64 k128 s4, exact)  M8 0.0637 (a16 n64 k128 s4, exact)  M10 0.0637 (a16 n64 k128 s8, exact)  M15 0.0636 (a16 n64 k128 s8, exact)  M16 0.0705 (a16 n128 k64 s1)  M20 0.0639 (a16 n64 k64 s8, exact)  M25 0.0639 (a16 n64 k64 s8, exact)  M32 0.0705 (a16 n128 k64 s1)  M40 0.0637 (a16 n64 k64 s4, exact)
  up: M1 0.0515 (a16 n64 k64 s1, exact)  M2 0.0512 (a16 n64 k64 s1, exact)  M3 0.0512 (a16 n64 k64 s1, exact)  M4 0.0512 (a16 n64 k64 s1, exact)  M5 0.0511 (a16 n64 k64 s1, exact)  M8 0.0507 (a16 n64 k64 s1, exact)  M10 0.0508 (a16 n128 k64 s1, exact)  M15 0.0513 (a16 n64 k64 s1, exact)  M16 0.0737 (quantized)  M20 0.0516 (a16 n64 k64 s1, exact)  M25 0.0517 (a16 n64 k64 s1, exact)  M32 0.0730 (quantized)  M40 0.0517 (a16 n64 k64 s1, exact)

== HC norm/mix kernels as served (b12x.norm.hyperconnection, capacity 8192, seed [seed hit]): ms per call
op                       1        5       40     1264     3024     6048     8192   us/token at 6048
grouped_rmsnorm     0.0320   0.0319   0.0317   0.2348   0.5545   1.1075   1.4949   0.18
scaled_silu         0.0328   0.0324   0.0326   0.0325   0.0325   0.0330   0.0335   0.01
gate_mean           0.0318   0.0315   0.0315   0.2623   0.6181   1.2285   1.6622   0.20
combine             0.0307   0.0305   0.0305   0.2729   0.6538   1.3056   1.8129   0.22
combine_norm        0.0412   0.0411   0.0410   0.4897   1.1499   2.2765   3.0861   0.38

== anchor: control down+up = 41.8 us/token over [6048, 6048, 3024, 1264] vs k81 dense-HC class 48.7 (ratio 0.86); ANCHOR=OK (OK = within 15%)
== projection: seconds saved per cold 16K prompt = sum over [6048, 6048, 3024, 1264] of (control - variant) ms x 96 down calls per step, k81-scaled headline (raw beside it)
  deq-pc: +0.296 s (+3.8% of 7.71 s), +0.254 s raw
  deq-cached: +0.343 s (+4.5% of 7.71 s), +0.295 s raw
  deq-b16: +0.311 s (+4.0% of 7.71 s), +0.267 s raw

== numerics gate per variant: deq-pc=PASS deq-cached=PASS deq-b16=PASS deq-up=FAIL
PREDICT=GO deq-pc +0.296 s (+3.8%)
VERDICT setup=1x served=hit anchor=OK predict=GO deq-pc=PASS deq-cached=PASS deq-b16=PASS deq-up=FAIL deq-pc=+0.296s

== P0 probe, 2x v2.1.0 image (TP=2 row: replicated HC, seed 225bf665)
setup 2x: seed 225bf665, 16K chunks [5728, 5728, 2864, 2064], cold 16K TTFT 5.60 s (k76 TP=2 cold 16K, another benchmark)
inputs: captured 16384 rows; weights: layer 5 attention HC, online MXFP8 g32

== down K10240 N336: full-op ms per call, x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails. Control: ctl-seed at capacity 8192, ctl (forced served config) above
cap      M           ctl-seed            deq-pc   control plan
 8192    256              0.072      0.161 x0.45~   a16 n128 k64 s1 [seed hit] -
 8192    512              0.136      0.189 x0.72~   a16 n128 k64 s1 [seed hit] -
 8192   1024              0.267      0.261 x1.02~   a16 n128 k64 s1 [seed hit] -
 8192   1264              0.334      0.284 x1.18=   a16 n128 k64 s1 [seed hit] -
 8192   2048              0.666      0.366 x1.82=   a16 n128 k64 s1 [seed hit] -
 8192   2064              0.689      0.378 x1.82=   a16 n128 k64 s1 [seed hit] -
 8192   2864              0.963      0.467 x2.06=   a16 n128 k64 s1 [seed hit] -
 8192   3024              0.943      0.464 x2.03=   a16 n128 k64 s1 [seed hit] -
 8192   4536              1.403      0.704 x1.99=   a16 n128 k64 s1 [seed hit] -
 8192   5728              1.783      0.830 x2.15=   a16 n128 k64 s1 [seed hit] -
 8192   6048              1.838      0.842 x2.18=   a16 n128 k64 s1 [seed hit] -
 8192   8192              2.480      1.175 x2.11=   a16 n128 k64 s1 [seed hit] -
 9072   9072                n/a            1.232!
12096  12096                n/a            1.592!

== up K320 N10240: full-op ms per call, x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails. Control: ctl-seed at capacity 8192, ctl (forced served config) above
cap      M           ctl-seed   control plan
 8192    256              0.077   quantized [seed hit] (64, 128)/BK128/u1
 8192    512              0.080   quantized [seed hit] (64, 128)/BK128/u1
 8192   1024              0.145   quantized [seed hit] (64, 128)/BK128/u1
 8192   1264              0.171   quantized [seed hit] (64, 128)/BK128/u1
 8192   2048              0.276   quantized [seed hit] (64, 128)/BK128/u1
 8192   2064              0.275   quantized [seed hit] (64, 128)/BK128/u1
 8192   2864              0.354   quantized [seed hit] (64, 128)/BK128/u1
 8192   3024              0.396   quantized [seed hit] (64, 128)/BK128/u1
 8192   4536              0.540   quantized [seed hit] (64, 128)/BK128/u1
 8192   5728              0.681   quantized [seed hit] (64, 128)/BK128/u1
 8192   6048              0.708   quantized [seed hit] (64, 128)/BK128/u1
 8192   8192              0.996   quantized [seed hit] (64, 128)/BK128/u1

== decode/drafter rows (capacity 8192, exact-M regimes [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]): ms per call, plan; the job compares their bytes between the shipped and the arm image
  down: M1 0.0638 (a16 n64 k64 s8, exact)  M2 0.0630 (a16 n64 k64 s8, exact)  M3 0.0630 (a16 n64 k128 s8, exact)  M4 0.0630 (a16 n64 k64 s8, exact)  M5 0.0627 (a16 n64 k64 s8, exact)  M8 0.0631 (a16 n64 k64 s8, exact)  M10 0.0634 (a16 n64 k64 s8, exact)  M15 0.0626 (a16 n64 k64 s8, exact)  M16 0.0703 (a16 n128 k64 s1)  M20 0.0634 (a16 n64 k64 s8, exact)  M25 0.0630 (a16 n64 k64 s8, exact)  M32 0.0703 (a16 n128 k64 s1)  M40 0.0631 (a16 n64 k64 s4, exact)  M45 0.0627 (a16 n64 k64 s4, exact)  M60 0.0633 (a16 n64 k64 s8, exact)  M80 0.0738 (quantized, exact)
  up: M1 0.0509 (a16 n64 k64 s1, exact)  M2 0.0511 (a16 n64 k64 s1, exact)  M3 0.0514 (a16 n64 k64 s1, exact)  M4 0.0514 (a16 n64 k64 s1, exact)  M5 0.0511 (a16 n128 k64 s1, exact)  M8 0.0512 (a16 n64 k64 s1, exact)  M10 0.0512 (a16 n128 k64 s1, exact)  M15 0.0512 (a16 n64 k64 s1, exact)  M16 0.0726 (quantized)  M20 0.0514 (a16 n128 k64 s1, exact)  M25 0.0514 (a16 n64 k64 s1, exact)  M32 0.0724 (quantized)  M40 0.0515 (a16 n64 k64 s1, exact)  M45 0.0514 (a16 n64 k64 s1, exact)  M60 0.0724 (quantized, exact)  M80 0.0732 (quantized, exact)

== anchor: control down+up = 42.2 us/token over [5728, 5728, 2864, 2064]; ANCHOR=N/A (k81 is a 1x profile)
== projection: seconds saved per cold 16K prompt = sum over [5728, 5728, 2864, 2064] of (control - variant) ms x 96 down calls per step, raw headline
  deq-pc: +0.260 s (+4.7% of 5.60 s)

== numerics gate per variant: deq-pc=PASS (deq-pc: no control at c12096,c9072, not judged there)
PREDICT=GO deq-pc +0.260 s (+4.7%)
VERDICT setup=2x served=hit anchor=N/A predict=GO deq-pc=PASS deq-pc=+0.260s

== P1 GPU unit tests in the arm image
    tile_sched_params = utils.PersistentTileSchedulerParams(

test_blockscaled_a16_dequant.py::test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below
test_blockscaled_a16_dequant.py::test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below
test_blockscaled_a16_dequant.py::test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below
test_blockscaled_a16_dequant.py::test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below
  /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir_helpers/op.py:294: DeprecationWarning: FastDivmodDivisor is deprecated in favor of cute.FastDivmodDivisorV2 / cute.fast_divmod_create_divisor_v2. V2 additionally carries the scalar divisor across kernel boundaries (2 MLIR values per object instead of 1), so '.divisor' is readable inside kernels; arithmetic (divmod, //, %) is unchanged.
    res_or_list = opFunc(*args, **kwargs, loc=loc)

test_blockscaled_a16_dequant.py::test_plan_regimes_routes_large_rows_through_bf16_and_keeps_the_kernel_below
  /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/utils/static_persistent_tile_scheduler.py:292: DeprecationWarning: Migrated to examples/CuTeDSL/helpers/static_persistent_tile_scheduler.py (BSD-3). The wheel copy will be removed in a future release.
    new_params = PersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 6 warnings in 6.53s

== P1 arm image vs shipped (ctl-seed and decode rows through the patched dispatch)
variant	shape	m	old_ms	new_ms	x	same_bytes	cos	maxrel	err_old	err_new	same_as_old_deq-pc	new_lowering
ctl-seed	down	256	0.0721	0.0719	1.00	yes	1.0000020	0.000e+00	0.001659	0.001659	no	bf16-deq>=1024
ctl-seed	down	512	0.1359	0.1361	1.00	yes	1.0000010	0.000e+00	0.001664	0.001664	no	bf16-deq>=1024
ctl-seed	down	1024	0.2670	0.2669	1.00	yes	1.0000007	0.000e+00	0.00166	0.00166	no	bf16-deq>=1024
ctl-seed	down	1264	0.3367	0.2883	1.17	yes	1.0000007	0.000e+00	0.00168	0.00168	yes	bf16-deq>=1024
ctl-seed	down	2048	0.6736	0.3653	1.84	yes	1.0000019	0.000e+00	0.001646	0.001646	yes	bf16-deq>=1024
ctl-seed	down	2064	0.6865	0.3817	1.80	yes	1.0000010	0.000e+00	0.001673	0.001673	yes	bf16-deq>=1024
ctl-seed	down	2864	0.9252	0.4715	1.96	yes	1.0000006	0.000e+00	0.001657	0.001657	yes	bf16-deq>=1024
ctl-seed	down	3024	0.9460	0.4865	1.94	yes	0.9999998	0.000e+00	0.001656	0.001656	yes	bf16-deq>=1024
ctl-seed	down	4536	1.3910	0.7132	1.95	yes	1.0000017	0.000e+00	0.001672	0.001672	yes	bf16-deq>=1024
ctl-seed	down	5728	1.8501	0.8641	2.14	yes	1.0000007	0.000e+00	0.001653	0.001653	yes	bf16-deq>=1024
ctl-seed	down	6048	1.9215	0.8673	2.22	yes	1.0000005	0.000e+00	0.001666	0.001666	yes	bf16-deq>=1024
ctl-seed	down	8192	2.5626	1.1807	2.17	yes	1.0000020	0.000e+00	0.001665	0.001665	yes	bf16-deq>=1024
ctl-seed	up	256	0.0759	0.0748	1.01	yes	0.9998292	0.000e+00	0.017936	0.017936	-	(64, 128)/BK128/u1
ctl-seed	up	512	0.0782	0.0771	1.01	yes	0.9998421	0.000e+00	0.018133	0.018133	-	(64, 128)/BK128/u1
ctl-seed	up	1024	0.1404	0.1383	1.02	yes	0.9998248	0.000e+00	0.017596	0.017596	-	(64, 128)/BK128/u1
ctl-seed	up	1264	0.1714	0.1692	1.01	yes	0.9998301	0.000e+00	0.018099	0.018099	-	(64, 128)/BK128/u1
ctl-seed	up	2048	0.2728	0.2702	1.01	yes	0.9998145	0.000e+00	0.017589	0.017589	-	(64, 128)/BK128/u1
ctl-seed	up	2064	0.2730	0.2767	0.99	yes	0.9998205	0.000e+00	0.018235	0.018235	-	(64, 128)/BK128/u1
ctl-seed	up	2864	0.3541	0.3487	1.02	yes	0.9998321	0.000e+00	0.016682	0.016682	-	(64, 128)/BK128/u1
ctl-seed	up	3024	0.3910	0.3971	0.98	yes	0.9998276	0.000e+00	0.017657	0.017657	-	(64, 128)/BK128/u1
ctl-seed	up	4536	0.5392	0.5383	1.00	yes	0.9998332	0.000e+00	0.016939	0.016939	-	(64, 128)/BK128/u1
ctl-seed	up	5728	0.6680	0.6906	0.97	yes	0.9998274	0.000e+00	0.016487	0.016487	-	(64, 128)/BK128/u1
ctl-seed	up	6048	0.7229	0.7266	0.99	yes	0.9998274	0.000e+00	0.016917	0.016917	-	(64, 128)/BK128/u1
ctl-seed	up	8192	0.9377	1.0004	0.94	yes	0.9998312	0.000e+00	0.017324	0.017324	-	(64, 128)/BK128/u1
dec	down	1	0.0650	0.0664	0.98	yes	1.0000000	0.000e+00	0.002104	0.002104	-	exact -
dec	down	2	0.0639	0.0646	0.99	yes	1.0000000	0.000e+00	0.001907	0.001907	-	exact -
dec	down	3	0.0639	0.0647	0.99	yes	1.0000002	0.000e+00	0.001811	0.001811	-	exact -
dec	down	4	0.0636	0.0646	0.98	yes	1.0000004	0.000e+00	0.001783	0.001783	-	exact -
dec	down	5	0.0640	0.0649	0.99	yes	1.0000005	0.000e+00	0.001756	0.001756	-	exact -
dec	down	8	0.0637	0.0644	0.99	yes	1.0000010	0.000e+00	0.0017	0.0017	-	exact -
dec	down	10	0.0637	0.0646	0.99	yes	1.0000013	0.000e+00	0.001716	0.001716	-	exact -
dec	down	15	0.0636	0.0643	0.99	yes	1.0000026	0.000e+00	0.001701	0.001701	-	exact -
dec	down	16	0.0705	0.0705	1.00	yes	1.0000030	0.000e+00	0.001695	0.001695	-	bf16-deq>=1024
dec	down	20	0.0639	0.0651	0.98	yes	1.0000030	0.000e+00	0.001709	0.001709	-	exact -
dec	down	25	0.0639	0.0650	0.98	yes	1.0000026	0.000e+00	0.001692	0.001692	-	exact -
dec	down	32	0.0705	0.0704	1.00	yes	1.0000033	0.000e+00	0.001692	0.001692	-	bf16-deq>=1024
dec	down	40	0.0637	0.0648	0.98	yes	1.0000044	0.000e+00	0.001681	0.001681	-	exact -
dec	up	1	0.0515	0.0521	0.99	yes	1.0000036	0.000e+00	0.001681	0.001681	-	exact -
dec	up	2	0.0512	0.0526	0.97	yes	1.0000077	0.000e+00	0.001659	0.001659	-	exact -
dec	up	3	0.0512	0.0524	0.98	yes	1.0000048	0.000e+00	0.00166	0.00166	-	exact -
dec	up	4	0.0512	0.0527	0.97	yes	0.9999998	0.000e+00	0.001655	0.001655	-	exact -
dec	up	5	0.0511	0.0524	0.98	yes	0.9999955	0.000e+00	0.001655	0.001655	-	exact -
dec	up	8	0.0507	0.0520	0.98	yes	0.9999937	0.000e+00	0.001651	0.001651	-	exact -
dec	up	10	0.0508	0.0525	0.97	yes	0.9999928	0.000e+00	0.001651	0.001651	-	exact -
dec	up	15	0.0513	0.0522	0.98	yes	0.9999664	0.000e+00	0.001648	0.001648	-	exact -
dec	up	16	0.0737	0.0744	0.99	yes	0.9999654	0.000e+00	0.017756	0.017756	-	(64, 128)/BK128/u1
dec	up	20	0.0516	0.0524	0.98	yes	0.9999564	0.000e+00	0.001649	0.001649	-	exact -
dec	up	25	0.0517	0.0525	0.98	yes	0.9999516	0.000e+00	0.001649	0.001649	-	exact -
dec	up	32	0.0730	0.0748	0.98	yes	0.9999455	0.000e+00	0.018309	0.018309	-	(64, 128)/BK128/u1
dec	up	40	0.0517	0.0527	0.98	yes	0.9999447	0.000e+00	0.00165	0.00165	-	exact -
XCHECK=PASS
```

Results: {RESULTS_URL}
