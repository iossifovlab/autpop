from scipy.stats import norm
import numpy as np

MXP = 4
vls = norm.rvs(0.235123, 0.001, size=10)


def method1(vls, MXP):
    for p in range(MXP, -1, -1):
        sts = {f'%.{p}f' % v for v in vls}
        if len(sts) == 1:
            r, = sts
            return r


def method2(vls, MXP):
    mn = np.mean(vls)
    for p in range(MXP, -1, -1):
        mvs = f'%.{p}f' % mn
        mv = float(f'%.{p}f' % mn)
        md = max([abs(mv-v) for v in vls])
        dst_cutoff = (10**-(p-1)) / 2
        print("\t", p, mvs, mv, md, dst_cutoff)
        if md < dst_cutoff:
            return mvs


print(method1(vls, MXP), method2(vls, MXP), np.mean(vls), vls)
