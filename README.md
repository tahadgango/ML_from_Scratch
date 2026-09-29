# Linear Regression from Scratch: Gradient-Descent Experiments

This project implements linear regression with NumPy and studies how optimization changes when updates use one example at a time or a small mini-batch. It also uses a full-batch learning-rate reference and a normal-equation least-squares baseline. The focus is not only on fitting a model: it is on building a sound experiment around it—separate train/validation/test data, learning-rate search, early stopping, multiple random seeds, result aggregation, and diagnostic plots.

The two notebooks apply the same workflow to the scikit-learn **Diabetes** and **California Housing** regression datasets. They implement the core algorithms directly rather than using scikit-learn's `LinearRegression` for fitting. The normal equation is included as a least-squares reference point.

## Project goals

- Implement the linear model, squared-error loss, gradients, and optimizers from scratch.
- Compare mini-batch gradient descent (batch size 32) with single-example stochastic gradient descent (SGD).
- Search a logarithmic grid of learning rates and quantify seed-to-seed variability.
- Select hyperparameters using validation data—not test data.
- Use learning curves and residual plots to understand behavior that a single score can conceal.

## Datasets

| Dataset | Size / features | Target | Why it is useful here |
| --- | ---: | --- | --- |
| Diabetes | 442 rows, 10 features | Quantitative disease-progression measure | A small-data setting where each split is consequential and stochastic variation is more visible. |
| California Housing | 20,640 rows, 8 features before the notebook's target-cap cleanup | Median house value (in units of $100,000) | A much larger, real-world regression dataset with geographic and structural effects that a linear model cannot fully express. |

For California Housing, the notebook excludes records at the maximum target value before splitting (`y < max(y)`), removing the dataset's capped target observations. Both notebooks standardize the input columns, then reserve 20% for testing and split the remaining 80% into 75% training and 25% validation. In overall terms, that is approximately **60% train / 20% validation / 20% test**.

> Important reproducibility note: the current notebooks compute feature means and standard deviations before splitting. For a strict no-leakage workflow, fit the scaler on the training partition only, then use those training statistics to transform validation and test data. The notebook results remain useful as optimization experiments, but this is the recommended revision for a final benchmark.

## Model and objective

For feature matrix $X$, weights $w$, and intercept $b$, the prediction is

$$
\hat{y} = Xw + b
$$

The notebooks minimize half the mean squared error:

$$
L(X, y; w, b) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

Its gradients are implemented as:

```python
errors = predict(X, w, b) - y
grad_w = X.T @ errors / len(X)
grad_b = np.mean(errors)
```

Each update follows:

$$
w \leftarrow w - \eta \nabla_w L,
\qquad
b \leftarrow b - \eta \nabla_b L
$$

where $\eta$ is the learning rate.

## Notebook walkthrough

### 1. Dataset exploration

Each notebook begins with visual exploratory analysis.

**California Housing** includes a geographic scatter/map view, feature-correlation heatmap, target distribution, median-income versus house-value plot, and structural-feature outlier checks. Together, these expose location effects, correlations, skew/capping in the target, and observations that may make a purely linear model imperfect.

**Diabetes** includes the target distribution, per-feature distributions, correlation matrix, feature-versus-target scatter plots, correlations with the target, and boxplots. This helps show scale, marginal relationships, and potential outliers before optimization begins.

These are descriptive plots—not model scores. They answer questions such as “what relationships and data issues should a linear model expect?”

### 2. Optimizers

The notebook contains four iterative fitting routines. It does not currently include a separate full-batch gradient-descent routine; full-batch behavior enters through the theoretical `eta_max` reference and the normal-equation baseline.

| Routine | Update data | Role |
| --- | --- | --- |
| `mini_batch_GD` | Shuffled batches of 32 examples | Fits on a supplied training set without validation stopping. |
| `mini_batch_GD_val` | Shuffled batches of 32 examples | Fits while tracking validation loss and restoring the best validation parameters. |
| `SGD` | One shuffled example | Fits on a supplied training set without validation stopping. |
| `SGD_val` | One shuffled example | Fits while tracking validation loss and restoring the best validation parameters. |

At every epoch, the data are shuffled. Mini-batch GD then updates once per batch; SGD updates once per observation. Both validation-aware routines initialize $w$ and $b$ randomly, record training and validation loss after each epoch, and use patience-based early stopping.

### 3. Early stopping and checkpointing

The validation optimizers use `patience = 20`. Whenever validation loss improves, they save a copy of the weights and bias and reset the patience counter. After 20 consecutive non-improving epochs, training stops. Crucially, the function returns the **saved best-validation parameters**, not merely the parameters from the final epoch.

This distinction explains an important pattern in noisy SGD curves: a trajectory can spike badly after it has already visited a good solution. The reported best validation loss and the raw final point on the curve are not necessarily the same thing.

### 4. Learning-rate range

The experiment uses:

```python
eta_max = 2 / max_eignvalue(X_train)
etas = np.logspace(-4, -1, 15)
```

`eta_max` is calculated from the largest eigenvalue of $X^T X / n$. It is a useful theoretical reference for full-batch optimization of this quadratic objective. The actual search uses 15 logarithmically spaced values from $10^{-4}$ through $10^{-1}$, which is appropriate because useful learning rates commonly differ by orders of magnitude.

The full-batch bound is **not** a guarantee that the same rate is stable for SGD or mini-batch updates. Stochastic gradients have additional variance; a rate that is reasonable under a deterministic full gradient may produce oscillation, overflow, or divergence when updates are based on a small sample.

## `run_experiment`: the experiment algorithm

`run_experiment` is the reusable layer that makes the mini-batch and SGD comparisons fair. It accepts an optimizer with the validation-aware return signature and runs the following procedure.

1. **For each candidate learning rate $\eta$**, create containers for training loss, best validation loss, test loss, stopping epoch, weights, and bias.
2. **For each seed**—the notebooks use `(42, 123, 99)`—set NumPy's seed. This changes random initialization and the per-epoch shuffle order.
3. **Fit the validation-aware optimizer** on the training set. It returns the stopping epoch, best validation loss, complete validation and training histories, and the checkpointed best parameters.
4. **Evaluate the returned parameters** on the test set and store the result. This is useful for a diagnostic table, but the test result must not influence choice of $\eta$.
5. **Save one learning-curve image per $\eta$-seed run** when an output directory is supplied. The blue line is epoch-level training loss and the red line is epoch-level validation loss.
6. **Aggregate across seeds** with mean and standard deviation for training, validation, and test loss, plus the mean stopping epoch.
7. **Keep the parameters from the seed with the lowest validation loss** for that particular $\eta$. This is a representative checkpoint for plotting/evaluation; the learning-rate ranking itself is based on the mean validation loss.
8. **Sort the aggregate table by `mean val loss`** and take the first row as the selected learning rate.

In compact form:

```text
learning rate grid
  └─ for each η
       └─ for seeds 42, 123, 99
            initialize + shuffle → train with validation early stopping
            save best-validation checkpoint and learning curves
       aggregate mean ± standard deviation
  choose η with lowest mean validation loss
```

### How to read the aggregate table

| Column | Meaning |
| --- | --- |
| `mean train loss` | Mean of the last recorded training loss for each seed; it is not necessarily the minimum training loss. |
| `mean val loss` | Mean of each seed's best validation loss; this is the primary selection criterion. |
| `mean test loss` | Mean held-out test loss from the seed runs; report it descriptively, but do not select with it. |
| `std ... loss` | Variation across the three seeds; a measure of sensitivity to initialization and shuffling. |
| `avg # of epoch` | Mean epoch at which training stopped (or the 1,000-epoch cap was reached). |

The appropriate default selection rule is:

```python
best_eta = results_table.loc[results_table["mean val loss"].idxmin(), "eta"]
```

Standard deviation is a **secondary stability diagnostic**, not a hard constraint by default. If two rates have practically indistinguishable mean validation loss, the lower-variance one is often the more reliable choice. A rate with the smallest standard deviation alone may simply be consistently poor.

## Evaluation and retraining workflow

The notebooks first evaluate the stored best-seed checkpoint on train, validation, and test partitions and plot test residuals. They also include an optional second stage:

```python
best_eta = df_result_sorted.iloc[0]["eta"]
loss_history, w, b = mini_batch_GD(X_train_val, y_train_val,
                                   eta=best_eta, epochs=epochs)
final_test_loss = loss(X_test, y_test, w, b)
```

The cleaner final workflow is:

1. Use train/validation data to choose $\eta$ and the stopping policy.
2. Lock those choices.
3. Retrain a fresh model on **train + validation** (80% of the data).
4. Evaluate once on the untouched test set.

Retraining is worthwhile because the selected configuration gets more fitting data. Do not call the checkpoint learned on only the training split the final model if the goal is a final 80%/20% evaluation. Conversely, do not repeatedly inspect test results while tuning rates, batch sizes, or epochs: that turns the test set into an implicit validation set.

For the retraining stage, define the epoch budget or a validation-free stopping rule before looking at the test result. Validation-based early stopping cannot directly operate after validation data have been folded into training.

## Visualizations and interpretation

| Visualization | Where it appears | What it shows / how to use it |
| --- | --- | --- |
| EDA distributions, correlations, scatter plots, and boxplots | Dataset analysis sections | Data scale, skew, pairwise association, potential nonlinearity, and outliers. These inform expectations before training. |
| California geographic map and income–price plot | California dataset analysis | Spatial structure and the influential relationship between income and target; neither is guaranteed to be linear. |
| Per-run learning curves | `plots/diabetes/` and `plots/california_housing/` | Training and validation loss over epochs for one $\eta$ and one seed. Look for fast convergence, under-training, widening train/validation separation, oscillation, and divergence. |
| Sorted learning-rate result table | Evaluation sections | Mean performance, seed-to-seed variation, and stopping time across the whole learning-rate grid. It is the basis for model selection. |
| Residual versus predicted-value scatter | “Best Pick” and optional retraining sections | Whether errors are centered near zero, whether spread grows with prediction size, nonlinear structure, and extreme predictions/outliers. |
| Train/validation/test $R^2$ | “Best Pick” and retraining outputs | Complementary explained-variance view. Compare partitions, but interpret differences in light of split randomness and dataset size. |

For a residual plot, the desirable baseline is a roughly patternless, zero-centered cloud with comparable vertical spread. Curvature, funnels, clusters, or isolated points are evidence that the linear model or its error assumptions are incomplete. In the California Housing discussion, an extreme negative prediction was observed in one residual plot; it should be investigated by locating that row, rather than silently dropping it from evaluation. A second, masked plot can help inspect the rest of the cloud, but the outlier must remain in reported metrics.

## Recorded experimental observations

The values below are notebook outputs for the shown split and seeds. They are snapshots, not universal properties of the datasets; rerunning with a different split or corrected scaling protocol can change them.

### Diabetes: a small-data, noisy learning-rate trade-off

For mini-batch GD, the recorded table selected $\eta = 0.1$ by mean validation loss: **1327.04 ± 11.69**, with mean test loss **1454.92 ± 39.62** and an average of about 50 epochs. The next rates had slightly worse mean validation loss but different stability profiles—for example, $\eta \approx 0.0373$ recorded **1353.02 ± 0.46** validation loss.

This does not mean that $0.1$ is universally preferable. Its individual mini-batch learning curves were visibly noisy: large stochastic updates could produce temporary loss spikes and later recover. The important methodological conclusion is:

> Choose by mean validation performance, then report standard deviation and learning curves to communicate reliability and optimization behavior.

The Diabetes SGD table selected $\eta \approx 0.0373$ by mean validation loss (**1305.50 ± 30.46**), but this was notably more variable across seeds. Smaller rates were more stable but, within the fixed training budget, generally had higher mean validation loss or slower progress. Since Diabetes has only 442 samples, a 20% validation set is small; split-specific choices can fluctuate. For a more formal small-data study, use repeated splits or cross-validation for hyperparameter selection.

### California Housing: stability limits show clearly

The recorded California mini-batch search selected $\eta \approx 0.0139$, with mean validation loss **0.214309 ± 0.000776**. In the recorded SGD search, the selected rate was much smaller, $\eta \approx 0.000439$, with mean validation loss **0.214327 ± 0.000720**. This is consistent with the fact that single-example SGD makes far noisier updates than mini-batches and generally requires a more conservative learning rate.

Large rates in the California outputs diverged dramatically, with overflow warnings and enormous or infinite losses. That is a useful result, not an inconvenience to hide: it directly demonstrates that the full-batch `eta_max` reference does not transfer as a stochastic stability guarantee.

The selected mini-batch checkpoint reported $R^2$ of about 0.50 (train), 0.55 (validation), and 0.58 (test); the optional retrained mini-batch model reported test $R^2$ about 0.58. These figures indicate meaningful signal, while the residual behavior and the nature of housing data also show the limits of a simple linear model. Location, capped values, heterogeneous variance, and nonlinear feature interactions are plausible sources of remaining error.

## Practical lessons

- **Mini-batch GD versus SGD:** Mini-batches average away some gradient noise, so their curves are usually smoother and their stable learning-rate region is wider. SGD updates more frequently and can move quickly, but it is more sensitive to shuffle order and needs smaller rates in the California experiment.
- **Learning rate is a trade-off:** A large rate may reach a good region quickly or yield a favorable best-validation checkpoint, but can oscillate, diverge, or vary more across seeds. A very small rate is often stable yet may fail to reach a good solution before the epoch limit.
- **Early stopping is part of the result:** The checkpoint is chosen at the lowest observed validation loss, not at the final training epoch. Therefore, inspect both the curves and the stopping epochs.
- **Mean and standard deviation answer different questions:** Mean validation loss measures average selection performance; standard deviation measures sensitivity to the random run. Report both as `mean ± std`.
- **Keep validation and test roles separate:** Validation selects learning rate and stopping behavior. Test data are for final, one-time generalization reporting.
- **Reproducibility needs more than fixed seeds:** The search fixes three seeds, which is a strong start. Also record the train/test split random state, scaler-fit partition, batch size, rate grid, epoch cap, patience, and library versions. In the current implementation, `np.random.seed(seed)` controls the initialization and permutations; the `random_state` arguments and locally created `rng` variables are not currently used by the update loops.

## Requirements and use

The notebooks use Python with NumPy, pandas, Matplotlib, Seaborn, and scikit-learn.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Run the Diabetes and California Housing notebooks from top to bottom. Learning curves are written below `plots/diabetes/` and `plots/california_housing/` when the corresponding experiment cell is executed.

## Conclusion

This project demonstrates that implementing linear regression is only part of the machine-learning task. A credible comparison requires repeatable splits, validation-driven hyperparameter selection, multiple seeds, uncertainty reporting, learning-curve inspection, residual diagnostics, and a final held-out evaluation. The experiments make the central trade-off tangible: aggressive stochastic optimization can look excellent at selected checkpoints, while conservative optimization is often smoother and more reproducible. Both perspectives belong in the final analysis.
