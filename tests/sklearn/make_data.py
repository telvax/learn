# Writes the data of the tests: a corpus of three topics (seeded), iris and diabetes from scikit-learn.
import csv, random, sys
from sklearn.datasets import load_iris, load_diabetes
out = sys.argv[1]
random.seed(7)
common = "the a of and to in is it for on with as was at by this that be are".split()
topics = {
    "sport": "goal match team player coach league score win season stadium ball referee".split(),
    "tech": "software computer data code network chip cloud model server app startup robot".split(),
    "food": "recipe cook kitchen flavor bread cheese wine dinner sauce garden fresh taste".split(),
}
others = [w for t in topics.values() for w in t]
with open(f"{out}/corpus.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    for i in range(900):
        label = list(topics)[i % 3]
        words = []
        for _ in range(random.randint(8, 25)):
            r = random.random()
            words.append(random.choice(topics[label]) if r < 0.35 else random.choice(others) if r < 0.55 else random.choice(common))
        if random.random() < 0.3:
            words.append(random.choice(["Great!", "café", "naïve", "x", "2026"]))
        w.writerow([" ".join(words), label])
iris = load_iris()
with open(f"{out}/iris.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    for x, y in zip(iris.data, iris.target):
        w.writerow([*x, iris.target_names[y]])
d = load_diabetes()
with open(f"{out}/diabetes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([*d.feature_names, "target"])
    for x, y in zip(d.data, d.target):
        w.writerow([*(f"{v:.17g}" for v in x), f"{y:.17g}"])
