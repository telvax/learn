# Compares the output of src/main.tvx with reference.py. The text features, Naive Bayes and least squares must be the same;
# logistic regression (a different solver of the same objective) and the forests (a different random generator) must be close.
import re, sys
mine = dict(l.split(" ", 1) for l in open(sys.argv[1]).read().splitlines() if " " in l)
theirs = dict(l.split(" ", 1) for l in open(sys.argv[2]).read().splitlines() if " " in l)
close = {"logistic-text": {"acc": 0.02, "loss": 0.005}, "logistic-iris": {"acc": 0.02, "loss": 0.005},
         "forest-iris": {"acc": -0.07}, "svc": {"acc": 0.01, "d00": 0.005, "d11": 0.005}, "forest-diabetes": {"r2": -0.08}}
bad = []
for key, want in theirs.items():
    got = mine.get(key)
    if got is None:
        bad.append(f"{key}: missing")
    elif key not in close:
        if got != want:
            bad.append(f"{key}:\n  telvax  {got}\n  sklearn {want}")
    else:
        g = dict(re.findall(r"(\w+)=([-\d.]+)", got)); w = dict(re.findall(r"(\w+)=([-\d.]+)", want))
        for m, tol in close[key].items():
            a, b = float(g[m]), float(w[m])
            ok = abs(a - b) <= tol if tol > 0 else a >= b + tol
            print(f"{key} {m}: telvax {a} sklearn {b} {'ok' if ok else 'TOO FAR'}")
            if not ok:
                bad.append(f"{key} {m}")
print("\n".join(bad) if bad else "same as scikit-learn")
sys.exit(1 if bad else 0)
