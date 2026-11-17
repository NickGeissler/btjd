"""Divide a vorticity dataset [N, T, 64, 64, 1] by one global constant.

The training set is divided by its own std (saved to <out>_scale.npy); held-out
sets must reuse that constant via --scale-from.
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scale-from', default=None, help='*_scale.npy of the training set')
    a = p.parse_args()

    raw = np.load(a.data).astype(np.float32)
    scale = float(np.load(a.scale_from)) if a.scale_from else float(np.std(raw))
    print(f'global scale = {scale:.10f}', flush=True)
    raw /= scale
    np.save(a.out, raw)
    np.save(a.out.rsplit('.npy', 1)[0] + '_scale.npy', np.array(scale))
    print(f'saved {a.out}', flush=True)


if __name__ == '__main__':
    main()
