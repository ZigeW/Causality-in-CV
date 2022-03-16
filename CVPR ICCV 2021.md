## CVPR 2021

### Causal Attention for Vision-Language Tasks

https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Causal_Attention_for_Vision-Language_Tasks_CVPR_2021_paper.pdf

propose Causal Attention (CATT) based on front-door adjustment principle that does not require assumption of any observed confounder

![1647256897315](https://github.com/ZigeW/Causality-in-CV/raw/main/images/1647256897315.png)

- attention mechanism as a front-door casual graph
  $$
  P(Y|X)=\underbrace{\sum_{z}P(Z=z|X)}_{IS-sampling}P(Y|Z=z)
  $$
  IS-sampling - In-Sample sampling since z comes from the current input x

- front-door adjustment - intervene on Z
  $$
  P(Y|do(X))=\underbrace{\sum_{z}P(Z=z|X)}_{IS-sampling}\underbrace{\sum_{x}P(X=x)}_{CS-sampling}P(Y|Z=z,X=x)
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