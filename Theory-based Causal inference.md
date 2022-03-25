## ITE Estimation

### $\bigstar$ ITE function

$$
\mathbb{E}[Y_1-Y_0|x]=\mathbb{E}[Y_1|x]-\mathbb{E}[Y_0|x]
$$

### $\bigstar$ Strong ignorability

strong ingorability $\Rightarrow$ ITE function to be identifiable
$$
\forall x, (Y_1,Y_0) \bot\bot t|x, \text{ and } 0<p(t=1|x)<1
$$
validity of strong ignorability cannot be assessed from data, and must be determined by domain knowledge and understanding of the causal relationships between var.

### $\bigstar$ Causal effect inference methods on estimation of ATE/ITE

* standard statistical methods
  * covariate (backdoor) adjustment, aka. G-computation formula
  * weighting methods - inverse propensity score weighting, etc.
* machine learning methods
  * tree-based methods
  * detect heterogeneous treatment effects
  * Neural nets

### Estimating individual treatment effect: generalization bounds and algorithms (ICML 2017)

http://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf

use two *Integral Probability Metric (IPM) measure* - *Maximum Mean Discrepancy (MMP)* and *Wasserstein distance*, to upper bound the additional variance in $p(t=1|x)$ and $p(t=0|x)​$ . Based on the idea of representation learning, jointly learn hypotheses for both treated and controlled on top of a representation which minimizes a weighted sum of loss and IPM distances.

![1648196397240](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1648196397240.png)

- predicting ITE from *observational data*, under the assumption of *strong ignorability* $\simeq$*no-hidden confounding*

- generalization-error bound - expected ITE estimation error of a representation is bounded by the standard generalization-error of that representation + the distance between the treated and control distributions induced by the representation

   (error of learning $Y_1$ and $Y_0$ + IPM term, when $t \bot\bot x$ , IPM is 0)

- bound $\rightarrow$  algo. learn a "balanced" representation such that the induced treated and control distribution look similar