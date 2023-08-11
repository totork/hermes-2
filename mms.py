from boutdata import collect as boutcollect
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import collections
import glob


def get_convergence(method):
    return 2


expected_failures = {"Div_a_Grad_perp_nonorthog(1, f)", "FV::Div_a_Grad_perp(1, f)"}

success = True
failed = collections.defaultdict(list)
failed2 = set()

linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 1))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


# normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
# def getColor():
#    color = colormap(normalize(mu))


def doit(path):
    def collect(var, mesh=0, path=path):
        # print(var, path, mesh)
        return boutcollect(
            var, path=path, prefix=f"BOUT.mesh_{mesh}", strict=True, info=False
        )

    r = 0  # os.system(f"build-cont-opt/hermes-mms -d {path} -q -q -q")
    if r:
        os.system(f"build-cont-opt/hermes-mms -d {path}")
        raise RuntimeError("bout++ failed")
    assert r == 0
    ana_default = {
        "R": lambda R, Z: 1 / R,
        "R²": lambda R, Z: R * 0 + 4,
        "sin(R)": lambda R, Z: np.cos(R) / R - np.sin(R),
        "sin(10*R)": lambda R, Z: 10 * np.cos(10 * R) / R - 1e2 * np.sin(10 * R),
        "sin(100*R)": lambda R, Z: 100 * np.cos(100 * R) / R - 1e4 * np.sin(100 * R),
        "sin(1000*R)": lambda R, Z: 1000 * np.cos(1000 * R) / R
        - 1e6 * np.sin(1000 * R),
        "sin(Z)*sin(R)": lambda R, Z: np.sin(Z) * np.cos(R) / R
        - 2 * np.sin(Z) * np.sin(R),
        "sin(Z)": lambda R, Z: -np.sin(Z),
        "sin(Z*10)": lambda R, Z: -np.sin(Z * 10) * 100,
        "sin(Z*100)": lambda R, Z: -np.sin(Z * 100) * 1e4,
        "sin(Z*1000)": lambda R, Z: -np.sin(Z * 1000) * 1e6,
        "Z": lambda R, Z: Z * 0,
    }
    ana = dict()
    ana["bracket(a, f)"] = {
        "R, Z": lambda R, Z: -1 / R,
        "R, R": lambda R, Z: 0 * R,
        "Z, R": lambda R, Z: 0 * R,
        "Z, Z": lambda R, Z: 0 * R,
        "sin(R), sin(Z)": lambda R, Z: -1 / R * np.cos(R) * np.cos(Z),
    }
    for a in 10, 100, 1000:
        ana["bracket(a, f)"][f"sin(R*{a}), sin(Z*{a})"] = (
            lambda R, Z, a=a: -1 / R * np.cos(a * R) * np.cos(a * Z) * a * a
        )
    ana["bracket(a, f, OLD)"] = ana["bracket(a, f)"]
    ana["FCI::Div_a_Grad_perp(a, f)"] = {
        "R, Z": lambda R, Z: 0 * R,
        "R, R": lambda R, Z: 0 * R + 2,
        "Z, R": lambda R, Z: Z / R,
        "Z, Z": lambda R, Z: 0 * Z + 1,
        "sin(R), sin(Z)": lambda R, Z: -np.sin(R) * np.sin(Z),
    }
    for a in 10, 100, 1000:
        ana["FCI::Div_a_Grad_perp(a, f)"][f"sin(R*{a}), sin(Z*{a})"] = (
            lambda R, Z, a=a: -np.sin(a * R) * np.sin(a * Z) * a * a
        )
    ana["FCI::dagp(f)"] = ana["FCI::Div_a_Grad_perp(a, f)"]

    def get_ana(method, func):
        try:
            dic = ana[method]
        except KeyError:
            dic = ana_default
        try:
            return dic[func]
        except:
            print(method, func)
            raise

    s = slice(2, -2), 1, slice(None)

    Rs = []
    m = 0
    while 1:
        try:
            Rs += [collect("R", m)]
            m += 1
        except OSError as e:
            # print(e)
            break
    mids = range(m)
    Zs = [collect("Z", m) for m in mids]
    assert len(Rs) == len(Zs)
    assert len(Rs) > 1
    toplot = []
    mmax = 0
    while 1:
        try:
            collect(f"out_{mmax}")
        except ValueError as e:
            break
        mmax += 1
    print(f"Checking {mmax} variables for {len(Zs)} meshes in dir {path}")
    for i in list(range(mmax)):
        l2 = []
        lst = []
        for m in mids:
            o = collect(f"out_{i}", m)
            attrs = o.attributes
            ops, inp = attrs["operator"], attrs["inp"]
            a = get_ana(ops, inp)(Rs[m], Zs[m])
            e = (o - a)[s]
            l2.append(np.sqrt(np.mean(e**2)))
            lst.append(Rs[m].shape[0] - 4)
        if not np.any(a):
            print(ops, inp)
            continue

        ord = []
        for i0 in range(len(l2) - 1):
            a, b = lst[i0 : i0 + 2]
            dx = b / a
            a, b = l2[i0 : i0 + 2]
            de = a / b
            ord += [np.log(de) / np.log(dx)]
        if not np.isclose(
            ord[-1], get_convergence(attrs["operator"]), atol=0.25, rtol=0
        ):
            print(i, ord, {k: v for k, v in attrs.items() if "_" not in k and v})

            global success, failed, failed2
            failed[ops].append(inp)
            if ops not in expected_failures:
                success = False
                failed2.add(ops)

        toplot.append((attrs["inp"], attrs["operator"], lst, l2))
        label = f'{attrs["inp"]} {attrs["operator"]}'
        with open(f"result_real_{i}.txt", "w") as f:
            f.write("real\n")
            f.write("{label}\n")
            f.write(" ".join([str(x) for x in lst]))
            f.write("\n")
            f.write(" ".join([str(x) for x in l2]))
            f.write("\n")
        # plt.plot([1 / x for x in lst], l2, label=label)
    toplot2 = dict()
    ass = set([x[0] for x in toplot])
    bss = set([x[0] for x in toplot])
    for a, b, c, d in toplot:
        toplot2[a] = []
        # toplot2[b] = []
    for a, b, c, d in toplot:
        toplot2[a].append((b, c, d))
        # toplot2[b].append((a, c, d))
    for k, vs in toplot2.items():
        plt.figure()
        for ab, c, d in vs:
            plt.plot(c, d, "x-", label=ab)
        plt.title(k)
        plt.legend()
        plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
    plt.show()
    # plt.figure()
    # plt.pcolormesh(Rs[m][s], Zs[m][s], o[s])
    # at = o.attributes
    # plt.title(
    #     " ".join([f"{k}:{at[k]}" for k in ["function", "operator"]])
    #     + f" {Rs[m].shape}"
    # )
    # plt.colorbar()
    # plt.savefig(f"/u/dave/Downloads/figs/mms_{m}_{i}.png")


if sys.argv[1:]:
    args = sys.argv[1:]
else:
    args = glob.iglob("mms*/")
for p in args:
    # print(p)
    doit(p)

if failed:
    print(failed)
if failed2:
    print(failed2)
sys.exit(0 if success else 1)
