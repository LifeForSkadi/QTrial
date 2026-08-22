# mqtbench_n15 — QTrial 评测结果

n_circuits=24, n_ok=24

- SWAP 相对 qiskit-o1 平均减少 **0.0%**
- SWAP 相对 qiskit-o3 平均减少 **-0.7%**

| circuit | n_qubits | method | swaps | twoq | depth | twoq_depth | fidelity | qiskit-o1_swaps | qiskit-o3_swaps | qiskit-o1_depth | qiskit-o3_depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mqtbench/ae_15 | 15 | hybrid_sabre_adopted | 81 | 474 | 998 | 474 | 0.028561518972863135 | 81 | 78 | 998 | 736 |
| mqtbench/bmw_quark_cardinality_15 | 15 | hybrid_sabre_adopted | 0 | 84 | 187 | 84 | 0.43713207250113056 | 0 | 0 | 187 | 147 |
| mqtbench/bv_15 | 15 | rl_multistart_routed | 3 | 16 | 33 | 16 | 0.8934142470898518 | 3 | 2 | 46 | 35 |
| mqtbench/dj_15 | 15 | hybrid_sabre_adopted | 8 | 38 | 88 | 38 | 0.7528844861496254 | 8 | 6 | 88 | 75 |
| mqtbench/dynamic_qft_15 | 15 | rl_multistart_routed | 0 | 0 | 30 | 0 | 0.8517881476360767 | 0 | 0 | 30 | 30 |
| mqtbench/ghz_15 | 15 | hybrid_sabre_adopted | 0 | 14 | 59 | 14 | 0.8774508243779107 | 0 | 0 | 59 | 45 |
| mqtbench/ghz_dynamic_15 | 15 | hybrid_sabre_adopted | 0 | 21 | 28 | 21 | 0.8218593137117908 | 0 | 0 | 28 | 28 |
| mqtbench/graphstate_15 | 15 | hybrid_sabre_adopted | 4 | 27 | 21 | 27 | 0.8407610310344253 | 4 | 2 | 21 | 20 |
| mqtbench/grover_15 | 15 | rl_multistart_routed | 102619 | 544657 | 1575220 | 544657 | 3.26e-322 | 102621 | 102090 | 1572671 | 1276100 |
| mqtbench/half_adder_15 | 15 | hybrid_sabre_adopted | 54 | 319 | 589 | 319 | 0.09608713255433002 | 54 | 43 | 589 | 465 |
| mqtbench/hhl_15 | 15 | rl_multistart_routed | 137 | 819 | 1918 | 819 | 0.0027190075506576057 | 143 | 133 | 1807 | 1255 |
| mqtbench/iqpe_15 | 2 | rl_multistart_routed | 0 | 28 | 282 | 28 | 0.6539687540089764 | 0 | 0 | 282 | 297 |
| mqtbench/qaoa_15 | 15 | hybrid_sabre_adopted | 78 | 434 | 633 | 434 | 0.06205090475786403 | 78 | 73 | 633 | 524 |
| mqtbench/qft_15 | 15 | hybrid_sabre_adopted | 62 | 396 | 607 | 396 | 0.05021010612601208 | 62 | 62 | 607 | 458 |
| mqtbench/qftentangled_15 | 15 | hybrid_sabre_adopted | 75 | 449 | 796 | 449 | 0.03485127753815516 | 75 | 66 | 796 | 503 |
| mqtbench/qnn_15 | 15 | hybrid_sabre_adopted | 0 | 14 | 62 | 14 | 0.8188650446516713 | 0 | 0 | 62 | 48 |
| mqtbench/qpeexact_15 | 15 | rl_multistart_routed | 67 | 411 | 802 | 411 | 0.04932274269560482 | 76 | 62 | 855 | 653 |
| mqtbench/qpeinexact_15 | 15 | rl_multistart_routed | 73 | 429 | 931 | 429 | 0.044706206179143455 | 76 | 58 | 861 | 698 |
| mqtbench/qwalk_15 | 15 | hybrid_sabre_adopted | 5399 | 27207 | 70669 | 27207 | 3.134764019599434e-81 | 5399 | 5275 | 70669 | 57616 |
| mqtbench/randomcircuit_15 | 15 | hybrid_sabre_adopted | 329 | 1789 | 2791 | 1789 | 5.008559629950835e-06 | 329 | 326 | 2791 | 2414 |
| mqtbench/vqe_real_amp_15 | 15 | hybrid_sabre_adopted | 0 | 42 | 77 | 42 | 0.6137547956244704 | 0 | 0 | 77 | 76 |
| mqtbench/vqe_su2_15 | 15 | hybrid_sabre_adopted | 0 | 42 | 80 | 42 | 0.5859793107710145 | 0 | 0 | 80 | 74 |
| mqtbench/vqe_two_local_15 | 15 | hybrid_sabre_adopted | 203 | 924 | 733 | 924 | 0.004369512730143385 | 203 | 195 | 733 | 576 |
| mqtbench/wstate_15 | 15 | hybrid_sabre_adopted | 0 | 28 | 78 | 28 | 0.7532192378911818 | 0 | 0 | 78 | 63 |
