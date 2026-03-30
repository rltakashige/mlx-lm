# type: ignore
"""Usage: python reproduce_arrayscache_leak.py --model PATH"""
import argparse
import ctypes


class MetalResourceCounter:
    def __init__(self):
        libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        metal = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Metal.framework/Metal"
        )
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        metal.MTLCreateSystemDefaultDevice()
        sel = libobjc.sel_registerName
        sel.restype, sel.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        cls = libobjc.objc_getClass
        cls.restype, cls.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        gm = libobjc.class_getInstanceMethod
        gm.restype, gm.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]
        gi = libobjc.method_getImplementation
        gi.restype, gi.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
        self._si = libobjc.method_setImplementation
        self._si.restype, self._si.argtypes = ctypes.c_void_p, [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dev = cls(b"IOGPUMetalDevice")
        self._am = gm(dev, sel(b"_addResource:"))
        self._rm = gm(dev, sel(b"_removeResource:"))
        self._oa, self._or = gi(self._am), gi(self._rm)
        IMP = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        oa, orr = IMP(self._oa), IMP(self._or)
        self.count = 0

        def on_add(s, sel, r):
            self.count += 1
            oa(s, sel, r)

        def on_rem(s, sel, r):
            self.count -= 1
            orr(s, sel, r)

        self._imps = (IMP(on_add), IMP(on_rem))
        self._si(self._am, ctypes.cast(self._imps[0], ctypes.c_void_p))
        self._si(self._rm, ctypes.cast(self._imps[1], ctypes.c_void_p))


counter = MetalResourceCounter()

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache, BatchKVCache
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--add-eval", action="store_true")
parser.add_argument(
    "--crash", action="store_true", help="Run until Metal resource limit crash"
)
args = parser.parse_args()

model, tokenizer = load(args.model)
gen = BatchGenerator(model=model, stop_tokens=set(), prefill_step_size=4096)
gen.insert(
    prompts=[tokenizer.encode("Hello"), tokenizer.encode("Hi")],
    max_tokens=[50000, 50000],
    samplers=[make_sampler(temp=0.7)] * 2,
)

num_steps = 50000 if args.crash else 5000
print_steps = max(1, num_steps // 10)
baseline = counter.count

for step in range(num_steps):
    gen.next()
    if args.add_eval:
        for c in gen.active_batch.cache:
            if isinstance(c, ArraysCache):
                if c.left_padding is not None:
                    mx.eval(c.left_padding)
                if c.lengths is not None:
                    mx.eval(c.lengths)
            elif isinstance(c, BatchKVCache):
                mx.eval(c.offset)
    if step % print_steps == 0:
        print(f"Step {step}: {counter.count} Metal resources")

growth = counter.count - baseline
print(
    f"Metal resources: {growth:+d} after {num_steps} steps ({growth / num_steps:+.1f}/step)"
)
