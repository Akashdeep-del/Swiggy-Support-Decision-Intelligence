Domain: Support

## How to run

Run these commands from the repository root on Python 3.10 or newer:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts use only the Python standard library and NumPy. `generate_data.py` creates `data/records.csv`. The four Part scripts regenerate their own text and CSV outputs. `common.py` contains shared helper functions and does not run a Part when imported.

## Part 1 findings

The estimated mean resolution time is **8.9382 minutes**, with a 95% interval from **8.3026 to 9.5738 minutes**. The test statistic is **2.8930**, which is greater than the two-tailed critical value 1.96, so H0 is rejected. A Type I error here would mean concluding that the true average resolution time differs from 8 minutes when it actually equals 8 minutes. The estimated gap from the planning assumption is **0.9382 minutes**; this is statistically detectable in this sample, but the operations team should judge whether a difference of under one minute changes staffing decisions.

The requested sample size is **1293**, compared with 800 available rows. Under the stated margin-of-error assumption, 800 records are not enough to reach the requested precision. Gradient descent and the closed-form regression give a slope gap of **0.0000**; repeated full-batch updates reduce the squared-error cost until the fitted line matches the closed-form solution.

The raw residual spread ratio is **3.7525**, showing a funnel because the upper fitted-value half has a much larger residual spread. After the log-log model, the ratio is **1.0396**, which is close to 1 and indicates that the spread is much more even. In the multiple regression, the predictors whose coefficients have absolute value above 0.05 are **ln(X4), X2, X3 and X6**. The adjusted R-squared rises from **0.6561** for the one-predictor log-log model to **0.7403** for the multiple model, so the additional predictors improve the in-sample fit under this measure.

## Part 2 findings

After 200 updates, the learning-rate losses from lowest to highest are **η = 1.0 (0.3703), η = 0.1 (0.3776), and η = 0.01 (0.5091)**. The small learning rate makes slower progress over the fixed number of updates, while the larger rate makes faster progress here; a rate that is too large can become unstable on other problems.

The three largest absolute logistic weights are **X3 (0.9013), X1 (-0.5554) and X2 (0.5505)**. The largest odds ratio is for X3: increasing SLA consumed ratio by one training standard deviation multiplies the modelled escalation odds by about **2.4629**, holding the other standardised predictors fixed.

The always-negative baseline accuracy is **0.7875**, while logistic accuracy at τ = 0.5 is **0.8042**. Accuracy alone is weak because the test set has many more non-escalations than escalations. Lowering τ from 0.5 to 0.3 raises recall from **0.4706** to **0.7451**, while precision falls from **0.5455** to **0.4634**.

At τ = 0.3, logistic regression has **F1 = 0.5714** and **recall = 0.7451**. KNN (K = 5) has **F1 = 0.3696** and **recall = 0.3333**. Logistic regression is stronger on both of these test measures for this split.

For the stated INR cost rule, I would use the logistic model with the training-selected threshold **τ = 0.1500**. Its test cost is **INR 6050**, compared with **INR 11800** at τ = 0.5 and **INR 20400** when nobody is flagged. The threshold is selected on training rows to avoid using test outcomes to tune a decision that is later evaluated on those same rows.

## Part 3 findings

The successive WCSS drops are **659.8282**, **108.0739**, **171.0575**, and **45.5466** for K = 2→3, 3→4, 4→5 and 5→6. The pinned elbow rule selects **K = 3** because the drop from K = 3 to 4 is less than half the preceding drop. The specified starting rows make this a reproducible solution, not necessarily the global minimum for every K.

The three K = 3 segments are:
- **A, low-friction tickets:** 322 rows, mean earlier tickets 1.9503, SLA ratio 0.2250, and escalation rate 0.0186. Keep these tickets in the normal queue and use self-service follow-up where appropriate.
- **B, repeat-contact and higher-value tickets:** 255 rows, mean earlier tickets 6.0706, account value 28.7192 thousand INR, and escalation rate 0.2353. Prioritise repeat-contact cases for an agent review.
- **C, high-SLA and high-escalation tickets:** 223 rows, mean SLA ratio 0.7118, satisfaction score 38.5215, and escalation rate 0.4664. Route tickets close to the SLA limit for early supervisor attention.

Segment **C** has the highest escalation rate at **0.4664**. The mean vector after standardisation is zero for each of the five predictors. The WCSS elbow is a clustering heuristic, not proof that three is the only meaningful grouping.

At the largest-gap cut, single linkage gives four clusters: **E1**, **E4**, **E5**, and **E2+E3+E6**. Complete linkage gives two clusters: **E1+E4+E5** and **E2+E3+E6**. Single linkage joins groups when any pair of members is close, so it can form a chain. Complete linkage considers the farthest pair, so it delays joining groups until their members are all relatively close.

PC1 is driven most strongly by **X1 (0.5317), X3 (-0.5187), X5 (0.4418) and X2 (-0.4100)**; the signs show the direction of each feature along this component. **Three components** are needed to retain at least 80% of the variance (0.8616). The mean PC1 scores for A, B and C are **1.7773**, **-0.7831** and **-1.6709**, respectively, which separates the three segment profiles along the first component.

## Part 4 findings

S4 (X1 ≤ 24) is the best candidate split by both information gain and Gini gain: its information gain is **0.1252** and its Gini gain is **0.0606**, the largest values among S1–S4. Growing a tree until every leaf is pure can memorise noise and small groups in the training data. A maximum-depth limit or a minimum number of samples per leaf can stop the tree from growing too far.

The logistic model has **F1 = 0.5714** and **recall = 0.7451**; hard voting has **F1 = 0.5344** and **recall = 0.6863**; soft voting has **F1 = 0.5507** and **recall = 0.7451**. Combining the models did not improve F1 over logistic regression on this test split, so the combination is not justified by these results.

After round 1 of AdaBoost, the two misclassified points each have weight 0.25, for a combined weight of **0.50**. Round 2 assigns h2 a larger coefficient (**0.8047**) than h1 (**0.5493**) because h2 has a lower weighted error after the first update.

**Model A: image-sized input.** The input is 28 × 28 × 1. The convolution and pooling output shapes are: conv1 **28 × 28 × 6** (156 parameters), max-pool **14 × 14 × 6** (0), conv2 **10 × 10 × 16** (2,416), max-pool **5 × 5 × 16** (0), flatten **400** values (0), dense1 **32** (12,832), and output **10** (330). The fully connected alternative has **13002** parameters in total; the CNN has **15734**. For the 128 × 128 × 3 comparison, the fully connected layer has **50332672** parameters, while the 3 × 3 convolution with 64 filters has **1792**.

**Model B: token sequence.** The recurrent parameter shapes are W_xh = **128 × 64**, W_hh = **128 × 128**, W_hy = **16 × 128**, b_h = **128**, and b_y = **16**, for **26768** parameters. The count does not grow with sequence length because the same recurrent weights are reused at each time step. For a 10-class output, use ReLU in hidden layers, softmax for the output and categorical cross-entropy as the loss. A very deep plain network can suffer vanishing gradients, so early layers receive weak learning signals. Skip connections provide a shorter path for information and gradients, making optimisation easier.

## Model-choice memo

**To: Head of Customer Support Operations**  
**Subject: Structured-ticket escalation model recommendation**

I recommend the logistic-regression model for a controlled pilot, using the training-selected alert threshold of **τ = 0.1500**. On the held-out test rows it achieved F1 **0.5714** at τ = 0.3 and recall **0.7451**; the cost-selected threshold produced a test cost of **INR 6050**. This is lower than the **INR 11800** cost at τ = 0.5 and the **INR 20400** cost when no tickets are flagged. The KNN model's F1 was **0.3696**, below logistic regression's **0.5714** at τ = 0.3. These comparisons support logistic regression as the simplest of the tested structured-data choices.

The evidence is limited because all 800 records are synthetic and performance is measured on one fixed split. Before operational use, the model would need validation on representative, appropriately governed real support data and monitoring for changes in ticket patterns. I would defer both image Model A and text Model B until the structured model has passed that validation; the current assignment only sizes those networks and does not train them.

## Limitations

- The 800 records are synthetic and do not represent real Swiggy customers, support policies or operational performance.
- The classification results use one fixed stratified train/test split; another split could produce different metrics and a different cost threshold.
- Only seven structured predictors are available, so relevant ticket context may be missing.
- The K-means solution depends on the pinned initial centroids and is not guaranteed to be a global optimum.
- The cost rule uses fixed assumed values of INR 50 per false alarm and INR 400 per missed escalation; actual operational costs may differ.
