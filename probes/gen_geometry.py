"""Generator geometry overrides (layer widths, depth) for the wide-world evaluations.
"""
import random


def make_id_gen_class(IdGen, layer_widths=None, depth=None):
    """Return IdGen itself, or a subclass whose gen_param forces layer widths
    (w0, w1) and/or depth d and redraws e under the same light rule with the
    new floor."""
    if not layer_widths and not depth:
        return IdGen
    w0, w1 = layer_widths if layer_widths else (None, None)

    class WideIdGen(IdGen):
        def gen_param(self):
            super().gen_param()
            if depth:
                self.d = depth
            if w0:
                self.w0, self.w1 = w0, w1
            min_e = (self.d - 1) * self.w0
            t0 = random.randint(min_e, self.max_edge)
            t1 = random.randint(min_e, self.max_edge)
            self.e = min(t0, t1)

    WideIdGen.__name__ = f"WideIdGen_{w0}_{w1}_d{depth}"
    return WideIdGen
