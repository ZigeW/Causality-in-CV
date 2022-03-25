# AAAI 2022

### A Causal Inference Look At Unsupervised Video Anomaly Detection 

https://www.aaai.org/AAAI22Papers/AAAI-37.LinX.pdf

propose a causal graph for VAD task then use backdoor adjustment; model ensemble for temporal context

![1647160664256](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647160664256.png)

- do(x) can not only intervene the direct parent of x, also the grandparents of x *(?)*
- for temporal context, use sliding window to create counterfactual feature, then add models together as ensembling

### A Causal Debiasing Framework for Unsupervised Salient Object Detection

https://www.aaai.org/AAAI22Papers/AAAI-108.LinX.pdf

observed two distribution biases in USOD task - contrast distribution bias (data-rich v.s. non-rich) and spatial distribution bias (center or not), propose a causal graph then use backdoor adjustment for the former and image-level weighting for the latter

![1647161659989](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647161659989.png)

- low-level visual appearance feature (from lower layer of backbone network) and high-level semantic feature (from higher layer) are concatenated together

### Causal Intervention for Subject-Deconfounded Facial Action Unit Recognition

https://www.aaai.org/AAAI22Papers/AAAI-399.ChenY.pdf

propose causal diagram and a plug-in causal intervention module (CIS)

![1647173158683](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647173158683.png)

- CIS - build a fixed dictionary of confounder using the training data which is updated after each epoch; approximate R as a weighted aggregation of all confounders using attention as weights

### Deconfounding Physical Dynamics with Global Causal Relation and Confounder Transmission for Counterfactual Prediction

https://www.aaai.org/AAAI22Papers/AAAI-3051.LiZ.pdf

propose global causal relation attention (GCRA) and confounder transmission structure (CTS)

- GCRA - encode temporal and spatial information by applying interframe attention and intraframe attention
- CTS - construct a causal graph using learned confounder information of GCRA as nodes and contact relations of objects as edges, a do-operation graph with same structure, concatenate two graphs together, then use GNN to propagate node and edge information

### Information-Theoretic Bias Reduction via Causal View of Spurious Correlation

https://arxiv.org/pdf/2201.03121.pdf

propose a structural causal model; use mutual information to measure the co-dependence of bias variables and features

![1647169213011](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647169213011.png)

- X - input image; C - context prior; Z - bias variable; Y - target label *(target label added in causal graph?)*
- add the mutual information of Z and F conditioned on Y as a loss regularizer

### Debiasing NLU Models via Causal Intervention and Counterfactual Reasoning

https://www.aaai.org/AAAI22Papers/AAAI-8503.TianB.pdf

causal graph for Natural Language Inference (NLI) task, use *counterfactual reasoning* instead of maximum likelihood to train network

![1647171173445](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647171173445.png)

- obtain the counterfactual situation (no-treatment condition) by not providing p/c/h

- total effect  = natural direct effect + total indirect effect (M as mediator)
  $$
  TE = Y_{x,M_{x}}-Y_{x*,M_{x*}}
  $$

  $$
  NDE=Y_{x,M_{x*}}-Y_{x*,M_{x*}}
  $$

### VACA: Designing Variational Graph Autoencoders for Causal Queries

https://www.aaai.org/AAAI22Papers/AAAI-12865.SanchezMartinP.pdf

model observational, interventional and counterfactual distribution using variational graph autoencoder

![1647246979948](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647246979948.png)

- encoder should not have hidden layer 
- decoder should have hidden layers no less than the longest direct path in causal graph - 1

### Deconfounded Visual Grounding

https://www.aaai.org/AAAI22Papers/AAAI-3671.HuangJ.pdf

https://github.com/psanch21/VACA

propose a causal graph; substitute the unobserved confounder with a confounder that can be learned as a dictionary; use simple language attention module to perform deconfouding

![1647248655334](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647248655334.png)

![1647248671173](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647248671173.png)

- do-intervention
  $$
  P(L|do(R),X)=E_{\hat{g} \sim \hat{G}}[P(L|R,X,\hat{g})]
  $$
  Normalized Weighted Geometric Mean
  $$
  P(L|do(R),X) \approx P(L|E_{\hat{g} \sim \hat{G}}[R,\hat{g}],X)
  $$

- backdoor adjustment
  $$
  E_{\hat{g} \sim \hat{G}}[R,\hat{g}]=\sum_{\hat{g} \sim \hat{G}}f(r,\hat{g})P(\hat{g})
  $$
  $f$ is feature fusion function (language attention module) and $P(\hat{g})$ is 1/n (n entries in dictionary)

- deconfounded visual grounding
  $$
  P(L=l|do(R=r),X=x) \approx P(l|\sum_{\hat{g}}f(R,\hat{g})P(\hat{g}) \oplus x)=P(l|r' \oplus x)
  $$
  

### Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification (Oral)

https://arxiv.org/pdf/2112.14380.pdf

use weighted sum of two Empirical Risk terms to learn a "real unbiased" model for long-tailed classification, and show the causal theory behind it.

- either overfit to "head" or "tail" leads to a biased model in long-tailed classification

- learn a balanced model and an imbalanced model, then sum up their ER as overall empirical risk minimization
  $$
  w^{imba}=\frac{(XE^{imba})^{\gamma}}{(XE^{imba})^{\gamma}+(XE^{ba})^{\gamma}}
  $$

  $$
  w^{ba}=1-w^{imba}
  $$

  $$
  R^{imba}(f)=-w^{imba}\sum_{i}y_ilogf_i(x)
  $$

  $$
  R^{ba}(f)=-w^{ba}\sum_{i}\hat{y}_ilogf_i(x)
  $$

  $$
  R(f)=R^{imba}(f)+R^{ba}(f)
  $$

  

- do-intervention and backdoor adjustment theory behind it
  $$
  \displaylines{R(f)=\sum_{(x,y)}\sum_{s\in\{0,1\}}L(y_s,f(x))\frac{P(X)}{P(x|S=s)}P(x,y,s)
  \\\ =\frac{1}{N}\sum_{(x,y)}[\underbrace{L(y_s=1,f(x))\frac{P(x)}{p(x|S=1)}}_ \text{Imbalanced Domain ER}
  \\\ + \underbrace{L(y_s=0,f(x))\frac{P(x)}{p(x|S=0)}}_ \text{Balanced Domain ER}]}
  $$

  $$
  w^{imba}\propto\frac{1}{P(S=1|x)}\propto(XE^{imba})^\gamma
  $$

  $$
  w^{ba}\propto\frac{1}{P(S=0|x)}\propto(XE^{ba})^\gamma
  $$

  

## CVPR 2021

### Causal Attention for Vision-Language Tasks

https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Causal_Attention_for_Vision-Language_Tasks_CVPR_2021_paper.pdf

propose Causal Attention (CATT) based on front-door adjustment principle that does not require assumption of any observed confounder

![1647256897315](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647256897315.png)

- attention mechanism as a front-door casual graph
  $$
  P(Y|X)=\underbrace{\sum_{z}P(Z=z|X)}_\text{IS-sampling}P(Y|Z=z)
  $$
  IS-sampling - In-Sample sampling since z comes from the current input x

- front-door adjustment - intervene on Z
  $$
  P(Y|do(X))=\underbrace{\sum_{z}P(Z=z|X)}_ \text{IS-sampling} \underbrace{\sum_{x}P(X=x)}_\text{CS-sampling}P(Y|Z=z,X=x)
  $$
  CS-sampling - Cross-Sample sampling since it comes from other samples (obtained by clustering embedded features into dictionaries)

### Towards Robust Classification Model by Counterfactual and Invariant Data Generation

https://openaccess.thecvf.com/content/CVPR2021/papers/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.pdf

generate counterfactual and factual data to improve the robustness of image classification models

![1647258419695](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647258419695.png)

### Counterfactual Zero-Shot and Open-Set Visual Recognition

https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Counterfactual_Zero-Shot_and_Open-Set_Visual_Recognition_CVPR_2021_paper.pdf

generate counterfactual sample by change class attributes and keep sample attributes unchanged to achieve *Counterfactual Faithfulness*, use *Consistency Rule* to perform seen/unseen binary classification  (if the counterfactual still looks like itself or not)

### Counterfactual VQA: A Cause-Effect Look at Language Bias

https://openaccess.thecvf.com/content/CVPR2021/papers/Niu_Counterfactual_VQA_A_Cause-Effect_Look_at_Language_Bias_CVPR_2021_paper.pdf

(similar as AAAI 2022 Debiasing NLU models...)

compute the total indirect effect as subtraction of total effect and natural direct effect (only given language or video)

![1647259795621](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647259795621.png)



## ICCV 2021

### Causal Attention for Unbiased Visual Recognition

https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Causal_Attention_for_Unbiased_Visual_Recognition_ICCV_2021_paper.pdf

propose a causal attention module (CaaM) that self-annotates the confounders by data partition, and overcome the over-adjustment in OOD settings by disentangling the confounder and mediator using complementary attention modules trained in adversarial style

![1647415964554](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647415964554.png)

- causal intervention by data partition - partition the training data into splits, each of which represents a confounder stratum
- Mini- and Maxi-game for better disentanglement of mediator and confounder

### Transporting Causal Mechanisms for Unsupervised Domain Adaptation

https://openaccess.thecvf.com/content/ICCV2021/papers/Yue_Transporting_Causal_Mechanisms_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.pdf

### Learning Causal Representation for Training Cross-Domain Pose Estimator via Generative Interventions

https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Learning_Causal_Representation_for_Training_Cross-Domain_Pose_Estimator_via_Generative_ICCV_2021_paper.html



# ACM MM 2021

### *Recovering the Unbiased Scene Graphs from the Biased Ones

https://dl.acm.org/doi/pdf/10.1145/3474085.3475297

tackle long-tailed problem in scene graph generation by removing *reporting bias* , which means the conspicuous classes (*e.g. on, in*) are more likely to be annotated than the inconspicuous ones (*e.g. parked on, covered in*).

![1647864291333](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647864291333.png)

