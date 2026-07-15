# Phase 0 Baseline Record

Run each workload with the same fixed prompt corpus and sampling parameters.
Alternate baseline and candidate runs, and capture clocks, power, temperature,
accepted output tokens/s, and inter-token latency (ITL).

| Run | Backend/revision | Workload | Concurrency | Accepted tok/s | ITL p50/p99 | GPU clock/power/temp | Profile |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| | | | | | | | |

For one representative decode/MTP step, use Nsight Systems to attribute time to
CPU/launch gaps, NVFP4 GEMMs, GDN, attention, MTP, sampling, and cache metadata
updates. Do not prioritize scheduler rewrites if CPU/launch time is below 3%.
