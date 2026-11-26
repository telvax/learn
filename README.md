# learn

Models for [Telvax](https://github.com/telvax/telvax), written in Telvax on `std.sparse` and `std.ml`. Each one gives the
results of scikit-learn for the same data: the tests run both and compare them.

```sh
tvpm add learn
```

```telvax
from learn.text import fit_tfidf, transform
from learn.bayes import fit_complement, predict

fn classify(train: List[Text], labels: List[Text], new: List[Text]) -> List[Text]:
    let v = fit_tfidf(train, 3000)
    let model = fit_complement(transform(v, train), labels, 1.0)
    predict(model, transform(v, new))
```

Every model is a plain value (a struct of lists and numbers), so `std.model.save(model, path)` writes it, together with
the vectorizer or the `std.ml.Features` it needs.

| Module | What it has | As scikit-learn's |
|---|---|---|
| `learn.text` | `tokenize`, `fit_counts(texts, max_features)`, `fit_tfidf(texts, max_features)`, `transform(v, texts) -> Sparse` | `CountVectorizer`, `TfidfVectorizer` with their defaults (`max_features` 0 keeps every word) |
| `learn.bayes` | `fit_multinomial(x, y, alpha)`, `fit_complement(x, y, alpha)`, `predict`, `scores` | `MultinomialNB`, `ComplementNB` |
| `learn.linear` | `fit_linear(rows, y)`, `predict_linear`; `fit_logistic(x, y, c, iterations)`, `predict_logistic`, `probabilities`; `sparse_rows(rows)` | `LinearRegression`; `LogisticRegression(C=c)` (multinomial, L-BFGS) |
| `learn.tree` | `fit_forest_classifier(rows, y, trees, seed)`, `fit_forest_regressor(rows, y, trees, seed)`, `predict_classes`, `predict_values` | `RandomForestClassifier`, `RandomForestRegressor` (CART, bootstrap, `sqrt` features for classes, all for numbers) |

Labels are `Text`; dense features are `List[List[Float]]`; text features are a `Sparse` matrix.

## How close to scikit-learn

`tests/sklearn` runs the same steps in Telvax (`tvpm run main`) and in scikit-learn 1.5.2 (`reference.py`), on a corpus of
three topics, iris and diabetes (`tests/data`, written by `make_data.py`). Then `compare.py` checks them:

* **The same, digit for digit:**
  * the vocabulary and its 20 most frequent words, the counts and the idf;
  * the predictions and accuracy of both Naive Bayes, on counts and on TF-IDF;
  * the intercept, the ten coefficients and R² of least squares.
* **The same to three decimals:** the accuracy and the log loss of logistic regression, on text and on iris. It is another
  solver of the same objective.
* **Close:** random forests use the generator of Telvax, so their trees are not scikit-learn's. Iris accuracy is 0.960 for
  both; diabetes R² is 0.401 against 0.430.

The TF-IDF values are stored as `f32` in `Sparse`, so their sums agree to about seven digits.

## Speed

Everything runs in the interpreter except the sparse products. Naive Bayes and the vectorizers handle tens of thousands of
documents in seconds. Forests are the slowest part, at about a second per tree for a few hundred rows; they are made for
the small tabular data of #251, not for millions of rows.

Licensed under MIT or Apache-2.0, at your choice.
