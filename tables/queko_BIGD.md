# queko_BIGD — QTrial 评测结果

n_circuits=30, n_ok=30

- SWAP 相对 qiskit-o1 平均减少 **-9.1%**
- SWAP 相对 qiskit-o3 平均减少 **-65.4%**

| circuit | n_qubits | method | swaps | twoq | depth | twoq_depth | fidelity | qiskit-o1_swaps | qiskit-o3_swaps | qiskit-o1_depth | qiskit-o3_depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_0 | 20 | rl_multistart_routed | 11 | 60 | 123 | 46 | 0.6962780770721958 | 12 | 6 | 133 | 93 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_1 | 20 | rl_multistart_routed | 10 | 61 | 136 | 52 | 0.6391273271945758 | 12 | 9 | 144 | 114 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_2 | 20 | rl_multistart_routed | 11 | 68 | 153 | 59 | 0.6675256217965944 | 13 | 4 | 166 | 93 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_3 | 20 | rl_multistart_routed | 8 | 47 | 119 | 43 | 0.7591143961734075 | 9 | 5 | 112 | 77 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_4 | 20 | rl_multistart_routed | 15 | 80 | 146 | 52 | 0.6166868423455452 | 12 | 8 | 156 | 127 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_5 | 20 | rl_multistart_routed | 6 | 41 | 53 | 22 | 0.7604145838823563 | 6 | 5 | 58 | 41 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_6 | 20 | rl_multistart_routed | 9 | 52 | 124 | 43 | 0.7397447848047362 | 10 | 7 | 121 | 80 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_7 | 20 | rl_multistart_routed | 7 | 50 | 93 | 32 | 0.718025370984069 | 12 | 7 | 90 | 60 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_8 | 20 | rl_multistart_routed | 13 | 68 | 128 | 49 | 0.6609417083207594 | 12 | 6 | 128 | 69 |
| queko_BIGD/20QBT_45CYC_.0D1_.1D2_9 | 20 | rl_multistart_routed | 7 | 40 | 71 | 30 | 0.7621992501627202 | 5 | 1 | 47 | 34 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_0 | 20 | rl_multistart_routed | 47 | 211 | 216 | 82 | 0.2702511783938988 | 41 | 22 | 212 | 148 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_1 | 20 | rl_multistart_routed | 20 | 118 | 93 | 33 | 0.4560505314241041 | 30 | 20 | 129 | 79 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_2 | 20 | rl_multistart_routed | 39 | 187 | 192 | 76 | 0.30879922966110573 | 35 | 24 | 170 | 146 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_3 | 20 | rl_multistart_routed | 25 | 141 | 139 | 50 | 0.3976674916423433 | 29 | 24 | 150 | 113 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_4 | 20 | rl_multistart_routed | 39 | 183 | 166 | 64 | 0.30782084500685686 | 35 | 16 | 169 | 85 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_5 | 20 | rl_multistart_routed | 35 | 169 | 226 | 85 | 0.33970960117275706 | 31 | 21 | 158 | 119 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_6 | 20 | rl_multistart_routed | 33 | 163 | 200 | 84 | 0.3386300666902953 | 30 | 21 | 150 | 100 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_7 | 20 | rl_multistart_routed | 39 | 175 | 237 | 94 | 0.33504283537755014 | 35 | 22 | 156 | 114 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_8 | 20 | rl_multistart_routed | 45 | 207 | 246 | 93 | 0.2717152046592793 | 38 | 25 | 214 | 150 |
| queko_BIGD/20QBT_45CYC_.0D1_.2D2_9 | 20 | rl_multistart_routed | 29 | 149 | 202 | 77 | 0.3756769245557972 | 27 | 14 | 147 | 93 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_0 | 20 | rl_multistart_routed | 56 | 263 | 287 | 121 | 0.1898619004393324 | 47 | 27 | 269 | 138 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_1 | 20 | rl_multistart_routed | 56 | 265 | 215 | 88 | 0.17202998046872123 | 45 | 35 | 208 | 148 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_2 | 20 | rl_multistart_routed | 55 | 266 | 246 | 96 | 0.1699891892623432 | 53 | 38 | 241 | 193 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_3 | 20 | rl_multistart_routed | 54 | 263 | 232 | 90 | 0.1783822186099847 | 61 | 41 | 242 | 175 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_4 | 20 | rl_multistart_routed | 64 | 289 | 277 | 113 | 0.15792799409109437 | 45 | 40 | 218 | 155 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_5 | 20 | rl_multistart_routed | 55 | 260 | 220 | 86 | 0.19178822533990028 | 44 | 34 | 207 | 142 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_6 | 20 | rl_multistart_routed | 59 | 282 | 236 | 99 | 0.15358709270584656 | 51 | 32 | 274 | 151 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_7 | 20 | rl_multistart_routed | 54 | 263 | 233 | 90 | 0.18275459937696442 | 43 | 30 | 204 | 166 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_8 | 20 | rl_multistart_routed | 47 | 236 | 214 | 79 | 0.19782793987559288 | 55 | 36 | 285 | 165 |
| queko_BIGD/20QBT_45CYC_.0D1_.3D2_9 | 20 | rl_multistart_routed | 38 | 181 | 197 | 74 | 0.3243457143901289 | 26 | 16 | 151 | 86 |
