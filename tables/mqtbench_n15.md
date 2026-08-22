# mqtbench_n15 — QTrial 评测结果

n_circuits=24, n_ok=24

- SWAP 相对 qiskit-o1 平均减少 **0.0%**
- SWAP 相对 qiskit-o3 平均减少 **-0.2%**

| circuit | n_qubits | method | swaps | twoq | depth | twoq_depth | fidelity | qiskit-o1_swaps | qiskit-o3_swaps | qiskit-o1_depth | qiskit-o3_depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mqtbench/ae_15 | 15 | rl_multistart_routed | 74 | 453 | 1045 | 453 | 0.024548915303380592 | 75 | 74 | 905 | 728 |
| mqtbench/bmw_quark_cardinality_15 | 15 | hybrid_sabre_adopted | 0 | 84 | 187 | 84 | 0.035288677594995085 | 0 | 0 | 187 | 147 |
| mqtbench/bv_15 | 15 | rl_multistart_routed | 3 | 16 | 27 | 16 | 0.8835485583958822 | 3 | 2 | 46 | 31 |
| mqtbench/dj_15 | 15 | hybrid_sabre_adopted | 8 | 38 | 88 | 38 | 0.20664110249646817 | 8 | 6 | 88 | 75 |
| mqtbench/dynamic_qft_15 | 15 | rl_multistart_routed | 0 | 0 | 30 | 0 | 0.8430364905434524 | 0 | 0 | 30 | 30 |
| mqtbench/ghz_15 | 15 | hybrid_sabre_adopted | 0 | 14 | 59 | 14 | 0.5900622322913729 | 0 | 0 | 59 | 45 |
| mqtbench/ghz_dynamic_15 | 15 | hybrid_sabre_adopted | 0 | 21 | 28 | 21 | 0.4201678496451373 | 0 | 0 | 28 | 28 |
| mqtbench/graphstate_15 | 15 | rl_multistart_routed | 3 | 24 | 22 | 24 | 0.862573774361033 | 3 | 2 | 17 | 19 |
| mqtbench/grover_15 | 15 | hybrid_sabre_adopted | 102393 | 543979 | 1574683 | 543979 | 4.4e-323 | 102393 | 102341 | 1574683 | 1274984 |
| mqtbench/half_adder_15 | 15 | rl_multistart_routed | 53 | 316 | 623 | 316 | 0.07503913880876652 | 54 | 51 | 618 | 468 |
| mqtbench/hhl_15 | 15 | hybrid_sabre_adopted | 143 | 837 | 1656 | 837 | 2.14998210367528e-10 | 143 | 135 | 1656 | 1279 |
| mqtbench/iqpe_15 | 2 | rl_multistart_routed | 0 | 28 | 282 | 28 | 0.661454671245046 | 0 | 0 | 282 | 297 |
| mqtbench/qaoa_15 | 15 | hybrid_sabre_adopted | 78 | 434 | 679 | 434 | 8.997208776488136e-08 | 78 | 67 | 679 | 563 |
| mqtbench/qft_15 | 15 | hybrid_sabre_adopted | 59 | 387 | 537 | 387 | 6.926284426204007e-05 | 59 | 60 | 537 | 477 |
| mqtbench/qftentangled_15 | 15 | hybrid_sabre_adopted | 68 | 428 | 616 | 428 | 1.2730832037579077e-05 | 68 | 66 | 616 | 573 |
| mqtbench/qnn_15 | 15 | hybrid_sabre_adopted | 0 | 14 | 62 | 14 | 0.4644325482354626 | 0 | 0 | 62 | 48 |
| mqtbench/qpeexact_15 | 15 | hybrid_sabre_adopted | 60 | 390 | 818 | 390 | 7.4623951336421e-08 | 60 | 59 | 818 | 528 |
| mqtbench/qpeinexact_15 | 15 | hybrid_sabre_adopted | 60 | 390 | 826 | 390 | 7.315514085489728e-08 | 60 | 59 | 826 | 714 |
| mqtbench/qwalk_15 | 15 | rl_multistart_routed | 5422 | 27276 | 70509 | 27276 | 1.1812381069349839e-83 | 5427 | 5342 | 70982 | 57469 |
| mqtbench/randomcircuit_15 | 15 | hybrid_sabre_adopted | 328 | 1786 | 2793 | 1786 | 1.7713820227607287e-27 | 328 | 332 | 2793 | 2189 |
| mqtbench/vqe_real_amp_15 | 15 | hybrid_sabre_adopted | 0 | 42 | 77 | 42 | 0.17070524777160948 | 0 | 0 | 77 | 76 |
| mqtbench/vqe_su2_15 | 15 | hybrid_sabre_adopted | 0 | 42 | 80 | 42 | 0.14822569722810067 | 0 | 0 | 80 | 74 |
| mqtbench/vqe_two_local_15 | 15 | hybrid_sabre_adopted | 215 | 960 | 900 | 960 | 5.690854757060229e-13 | 215 | 181 | 900 | 533 |
| mqtbench/wstate_15 | 15 | hybrid_sabre_adopted | 0 | 28 | 78 | 28 | 0.3187965946757109 | 0 | 0 | 78 | 63 |
