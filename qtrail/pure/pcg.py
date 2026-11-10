"""PCG64Mcg（rand_pcg 0.3 语义）：qiskit rust 平局裁决同款 PRNG。"""
MASK = (1 << 128) - 1
MULT = 0x2360ED051FC65DA44385DF649FCCF645


class Pcg64Mcg:
    def __init__(self, seed: int):
        self.state = (seed + 1) & MASK
        self.inc = 1
        self._step()

    def _step(self):
        self.state = (self.state * MULT + self.inc) & MASK

    def next_u64(self) -> int:
        s = self.state
        self._step()
        xorshifted = (((s >> 64) ^ s) & 0xFFFFFFFFFFFFFFFF)
        rot = (s >> 122) & 0x3F
        return (((xorshifted >> rot) | (xorshifted << (64 - rot)))
                & 0xFFFFFFFFFFFFFFFF) if rot else xorshifted

    def gen_range(self, n: int) -> int:
        zone = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self.next_u64()
            if v < zone:
                return v % n
