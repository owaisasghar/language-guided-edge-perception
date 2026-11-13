# Query latency vs index size (Milestone 6)

**Actual measurement**, `scripts/benchmark_query.py`: a real index of 809 tracks (4 simulated cameras, 2 minutes) enlarged with jittered copies; warm search (text embeddings cached), NumPy on the CPU (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics); 5 queries x 20 repeats.

| tracks | parse ms (median) | search ms (median / p90) | embedding storage per track |
|---|---|---|---|
| 1,000 | 0.04 | 1.4 / 5.0 | 8.0 KiB |
| 10,000 | 0.09 | 15.3 / 31.2 | 8.0 KiB |
| 100,000 | 0.11 | 149.2 / 298.3 | 8.0 KiB |
