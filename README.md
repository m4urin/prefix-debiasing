# Debiasing through Prefix Tuning

[Paper](https://github.com/m4urin/prefix-debiasing/blob/main/paper.pdf).

Abstract:

By providing the right prompts, language models can identify their own biases and adjust their predictions accordingly (Schick et al., 2021). This process also alters the model’s word embeddings, leading to our hypothesis that learning these prompts may decrease unwanted bias in word embeddings. In this study, we propose a method to mitigate bias in pre-trained language models (PLMs) through the use of Prefix-Tuning (Li and Liang, 2021), which is a technique for training prompts in continuous space for downstream tasks. The training objective to debias the model is based on contextual orthogonal training (Kaneko and Bollegala, 2021), which utilizes lists of gendered attributes and stereotypical words. The results demonstrate that this method can debias language models to a certain degree and performs similarly to fine-tuning methods using standard evaluation techniques. Additionally, the method is highly efficient, requiring only 0.4% of the model’s parameters to be trained and stored. However, it should be noted that the method does not perform well on downstream tasks when continuing to fine-tune on parameters other than the learned prompt. These findings suggest that prompts can have a significant impact on word embeddings and may inspire further research on their use for debiasing language models. This work comes with an extensive literature study on measuring and mitigating bias in language models and the challenges that come with it. 
