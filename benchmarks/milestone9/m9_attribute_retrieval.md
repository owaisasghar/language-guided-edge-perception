# Attribute retrieval on CityFlow-NL: "find all white SUVs" (zero-shot)

**Actual measurements**, 2026-11-15, `scripts/eval_attribute_retrieval.py`. CityFlow-NL annotated vehicles, one track each; labels = majority colour and type of the three descriptions (parsed; grey = silver); queries generated from labels with >= 3 vehicles; ranking from image embeddings only; attribute weight chosen on S02 by mAP, reported on S05; text weight 4.0.

S05: 337 vehicles (337 with a described colour, 332 with a described type), 35 queries.

| scoring | attribute weight | mAP | random mAP | P@1 (Rank-1) | P@5 | P@10 | R@10 | R@50 |
|---|---|---|---|---|---|---|---|---|
| description similarity only | 0.0 | 0.458 | 0.084 | 0.629 | 0.571 | 0.483 | 0.306 | 0.628 |
| + colour check | 0.1 | 0.475 | 0.084 | 0.657 | 0.589 | 0.483 | 0.308 | 0.643 |
| + colour and type check | 2.0 | 0.514 | 0.084 | 0.686 | 0.646 | 0.526 | 0.313 | 0.631 |

Paired bootstrap over queries (2,000 resamples), mAP difference and 95 % interval:

* + colour check minus description similarity only: +0.017 [+0.002, +0.032]
* + colour and type check minus + colour check: +0.039 [-0.002, +0.078]
* + colour and type check minus description similarity only: +0.056 [+0.011, +0.097]

By query family (+ colour and type check):

| family | queries | mAP | random mAP | P@1 | P@10 | R@50 |
|---|---|---|---|---|---|---|
| colour | 6 | 0.716 | 0.166 | 1.000 | 0.750 | 0.589 |
| type | 7 | 0.549 | 0.141 | 0.857 | 0.586 | 0.537 |
| colour + type | 22 | 0.449 | 0.043 | 0.545 | 0.445 | 0.672 |

Per query (+ colour and type check):

| query | correct | AP | P@10 |
|---|---|---|---|
| an SUV | 140 | 0.852 | 1.000 |
| a sedan | 108 | 0.905 | 1.000 |
| a gray vehicle | 93 | 0.704 | 0.900 |
| a white vehicle | 89 | 0.888 | 1.000 |
| a black vehicle | 83 | 0.714 | 0.700 |
| a red vehicle | 46 | 0.826 | 1.000 |
| a black SUV | 43 | 0.658 | 0.700 |
| a pickup truck | 42 | 0.942 | 1.000 |
| a gray sedan | 39 | 0.625 | 0.700 |
| a gray SUV | 33 | 0.416 | 0.700 |
| a white SUV | 33 | 0.675 | 0.800 |
| a white sedan | 26 | 0.609 | 0.600 |
| a black sedan | 24 | 0.798 | 0.900 |
| a red SUV | 19 | 0.582 | 0.900 |
| a blue vehicle | 18 | 0.278 | 0.400 |
| a van | 17 | 0.409 | 0.400 |
| a red sedan | 16 | 0.427 | 0.600 |
| a blue SUV | 12 | 0.280 | 0.300 |
| a white pickup truck | 12 | 0.697 | 0.800 |
| a hatchback | 11 | 0.329 | 0.300 |
| a station wagon | 9 | 0.313 | 0.300 |
| a gray pickup truck | 9 | 0.705 | 0.600 |
| a black pickup truck | 8 | 0.627 | 0.600 |
| a red pickup truck | 8 | 0.803 | 0.600 |
| a white van | 8 | 0.431 | 0.300 |
| a green vehicle | 6 | 0.883 | 0.500 |
| a minivan | 5 | 0.090 | 0.100 |
| a black van | 5 | 0.043 | 0.000 |
| a gray hatchback | 4 | 0.064 | 0.100 |
| a white station wagon | 4 | 0.058 | 0.000 |
| a blue pickup truck | 3 | 0.047 | 0.000 |
| a blue sedan | 3 | 0.185 | 0.100 |
| a gray minivan | 3 | 0.239 | 0.200 |
| a gray station wagon | 3 | 0.421 | 0.100 |
| a white hatchback | 3 | 0.482 | 0.200 |

Scoring latency: median 0.3 ms, p90 3.1 ms per query over 337 vehicles (CPU; text embedding on the GPU; gallery embeddings precomputed).
