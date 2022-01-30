# Reading-Group
This repository contains our review slides for various papers in the field of machine learning.

## Papers

**1. An Equivalence Between Private Classification and Online Prediction**
```
Bun, M., Livni, R., & Moran, S. (2020, November). An equivalence between private classification and online prediction. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS) (pp. 389-402). IEEE.
```
> **Abstract**
> 
> We prove that every concept class with finite Littlestone dimension can be learned by an (approximate) differentially-private algorithm. This answers an open question of Alon et al. (STOC 2019) who proved the converse statement (this question was also asked by Neel et al. (FOCS 2019)). Together these two results yield an equivalence between online learnability and private PAC learnability. We introduce a new notion of algorithmic stability called “global stability” which is essential to our proof and may be of independent interest. We also discuss an application of our results to boosting the privacy and accuracy parameters of differentially-private learners.
> 
> [paper](https://arxiv.org/abs/2003.00563)

**2. Decision-based adversarial attacks: Reliable attacks against black-box machine learning models**
```
Brendel, W., Rauber, J., & Bethge, M. (2017). Decision-based adversarial attacks: Reliable attacks against black-box machine learning models. arXiv preprint arXiv:1712.04248.
```
> **Abstract**
> 
> Many machine learning algorithms are vulnerable to almost imperceptible perturbations of their inputs. So far it was unclear how much risk adversarial perturbations carry for the safety of real-world machine learning applications because most methods used to generate such perturbations rely either on detailed model information (gradient-based attacks) or on confidence scores such as class probabilities (score-based attacks), neither of which are available in most real-world scenarios. In many such cases one currently needs to retreat to transfer-based attacks which rely on cumbersome substitute models, need access to the training data and can be defended against. Here we emphasise the importance of attacks which solely rely on the final model decision. Such decision-based attacks are (1) applicable to real-world black-box models such as autonomous cars, (2) need less knowledge and are easier to apply than transfer-based attacks and (3) are more robust to simple defences than gradient- or score-based attacks. Previous attacks in this category were limited to simple models or simple datasets. Here we introduce the Boundary Attack, a decision-based attack that starts from a large adversarial perturbation and then seeks to reduce the perturbation while staying adversarial. The attack is conceptually simple, requires close to no hyperparameter tuning, does not rely on substitute models and is competitive with the best gradient-based attacks in standard computer vision tasks like ImageNet.
> 
> [paper](https://arxiv.org/pdf/1712.04248.pdf)

**3. A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima**
```
Xie, Z., Sato, I., & Sugiyama, M. (2020). A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima. arXiv preprint arXiv:2002.03495.
```
> **Abstract**
> 
> Stochastic Gradient Descent (SGD) and its variants are mainstream methods for training deep networks in practice. SGD is known to find a flat minimum that often generalizes well. However, it is mathematically unclear how deep learning can select a flat minimum among so many minima. To answer the question quantitatively, we develop a density diffusion theory (DDT) to reveal how minima selection quantitatively depends on the minima sharpness and the hyperparameters. To the best of our knowledge, we are the first to theoretically and empirically prove that, benefited from the Hessian-dependent covariance of stochastic gradient noise, SGD favors flat minima exponentially more than sharp minima, while Gradient Descent (GD) with injected white noise favors flat minima only polynomially more than sharp minima. We also reveal that either a small learning rate or large-batch training requires exponentially many iterations to escape from minima in terms of the ratio of the batch size and learning rate. Thus, large-batch training cannot search flat minima efficiently in a realistic computational time.
> 
> [paper](https://arxiv.org/pdf/1712.04248.pdf)


**4. Differentially private fine-tuning of language models**
```
Yu, D., Naik, S., Backurs, A., Gopi, S., Inan, H. A., Kamath, G., ... & Zhang, H. (2021). Differentially private fine-tuning of language models. arXiv preprint arXiv:2110.06500.
```
> **Abstract**
> 
> We give simpler, sparser, and faster algorithms for differentially private fine-tuning of large-scale pre-trained language models, which achieve the state-of-the-art privacy versus utility tradeoffs on many standard NLP tasks. We propose a meta-framework for this problem, inspired by the recent success of highly parameter-efficient methods for fine-tuning. Our experiments show that differentially private adaptations of these approaches outperform previous private algorithms in three important dimensions: utility, privacy, and the computational and memory cost of private training. On many commonly studied datasets, the utility of private models approaches that of non-private models. For example, on the MNLI dataset we achieve an accuracy of 87.8% using RoBERTa-Large and 83.5% using RoBERTa-Base with a privacy budget of ϵ=6.7. In comparison, absent privacy constraints, RoBERTa-Large achieves an accuracy of 90.2%. Our findings are similar for natural language generation tasks. Privately fine-tuning with DART, GPT-2-Small, GPT-2-Medium, GPT-2-Large, and GPT-2-XL achieve BLEU scores of 38.5, 42.0, 43.1, and 43.8 respectively (privacy budget of ϵ=6.8,δ= 1e-5) whereas the non-private baseline is 48.1. All our experiments suggest that larger models are better suited for private fine-tuning: while they are well known to achieve superior accuracy non-privately, we find that they also better maintain their accuracy when privacy is introduced
> 
> [paper](https://arxiv.org/pdf/2110.06500.pdf)
