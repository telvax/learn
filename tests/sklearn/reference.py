# What scikit-learn prints for the same steps as src/main.tvx (run with scikit-learn 1.5.2).
import csv, sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score
data = sys.argv[1] if len(sys.argv) > 1 else "../data"
rows = list(csv.DictReader(open(f"{data}/corpus.csv")))
train, test = rows[:600], rows[600:]
X, y = [r["text"] for r in train], [r["label"] for r in train]
Xt, yt = [r["text"] for r in test], [r["label"] for r in test]
s = lambda v: f"{v:.6f}"
cv = CountVectorizer(); xc = cv.fit_transform(X); words = list(cv.get_feature_names_out())
print(f"counts {len(words)} {','.join(words[:8])} nnz={xc.nnz} total={s(xc.sum())}")
top = CountVectorizer(max_features=20).fit(X)
print(f"top20 {','.join(top.get_feature_names_out())}")
tv = TfidfVectorizer(); xt = tv.fit_transform(X)
print(f"tfidf total={xt.sum():.4f} row0={s(xt[0].sum())} idf0={s(tv.idf_[0])}")
xtest = cv.transform(Xt)
p = MultinomialNB(alpha=1.0).fit(xc, y).predict(xtest)
print(f"mnb acc={s(accuracy_score(yt, p))} {','.join(p[:30])}")
p = ComplementNB(alpha=1.0).fit(xc, y).predict(xtest)
print(f"cnb acc={s(accuracy_score(yt, p))} {','.join(p[:30])}")
p = MultinomialNB(alpha=0.1).fit(xt, y).predict(tv.transform(Xt))
print(f"mnb-tfidf acc={s(accuracy_score(yt, p))}")
from sklearn.svm import LinearSVC
tw = TfidfVectorizer(max_df=0.85, max_features=40, ngram_range=(1, 2), sublinear_tf=True); x2 = tw.fit_transform(X)
print(f"pipeline-tfidf {','.join(tw.get_feature_names_out())} total={x2.sum():.4f}")
print(f"lda-counts {','.join(CountVectorizer(max_df=0.85, stop_words='english').fit(X).get_feature_names_out())}")
svc = LinearSVC(random_state=42).fit(x2, y); xt2 = tw.transform(Xt); d = svc.decision_function(xt2)
print(f"svc acc={accuracy_score(yt, svc.predict(xt2)):.3f} d00={d[0][0]:.3f} d11={d[1][1]:.3f}")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, log_loss
d = list(csv.DictReader(open(f"{data}/diabetes.csv")))
names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
x = np.array([[float(r[n]) for n in names] for r in d]); yd = np.array([float(r["target"]) for r in d])
lr = LinearRegression().fit(x, yd)
print(f"linear intercept={lr.intercept_:.6f} coef={','.join(f'{v:.4f}' for v in lr.coef_)} r2={r2_score(yd, lr.predict(x)):.6f}")
m = LogisticRegression(C=1.0, max_iter=5000).fit(xc, y)
print(f"logistic-text acc={accuracy_score(yt, m.predict(xtest)):.3f} loss={log_loss(yt, m.predict_proba(xtest)):.3f}")
ir = list(csv.DictReader(open(f"{data}/iris.csv")))
xi = np.array([[float(r[n]) for n in ["sepal_length", "sepal_width", "petal_length", "petal_width"]] for r in ir]); yi = [r["species"] for r in ir]
m = LogisticRegression(C=1.0, max_iter=5000).fit(xi, yi)
print(f"logistic-iris acc={accuracy_score(yi, m.predict(xi)):.3f} loss={log_loss(yi, m.predict_proba(xi)):.3f}")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
tr = [i for i in range(len(yi)) if i % 3 != 0]; te = [i for i in range(len(yi)) if i % 3 == 0]
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(xi[tr], [yi[i] for i in tr])
print(f"forest-iris acc={accuracy_score([yi[i] for i in te], rf.predict(xi[te])):.3f}")
fr = RandomForestRegressor(n_estimators=30, random_state=42).fit(x[:342], yd[:342])
print(f"forest-diabetes r2={r2_score(yd[342:], fr.predict(x[342:])):.3f}")
from sklearn.decomposition import LatentDirichletAllocation
themes = {}
for name, words in {"sport": "goal match team player coach league score win season stadium ball referee", "tech": "software computer data code network chip cloud model server app startup robot", "food": "recipe cook kitchen flavor bread cheese wine dinner sauce garden fresh taste"}.items():
    for w in words.split():
        themes[w] = name
cvl = CountVectorizer(max_df=0.85, stop_words="english"); xl = cvl.fit_transform([r["text"] for r in rows])
lda = LatentDirichletAllocation(n_components=3, random_state=42).fit(xl)
names = cvl.get_feature_names_out(); found = []
for topic in lda.components_:
    top = [themes.get(names[i], "none") for i in topic.argsort()[::-1][:5]]
    found.append(top[0] if all(n == top[0] for n in top) else "mixed")
print(f"lda-topics {','.join(sorted(found))}")
