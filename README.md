<!-- omit in toc -->
# unsloth-fine-tuning

- [1. how to fine tune gpt-oss](#1-how-to-fine-tune-gpt-oss)
- [2. LoRA hyperparameters guide](#2-lora-hyperparameters-guide)
  - [2.1. what is LoRA?](#21-what-is-lora)
  - [2.2. key fine-tuning hyperparameters](#22-key-fine-tuning-hyperparameters)
  - [2.3. gradient accumulation and batch size equivalency](#23-gradient-accumulation-and-batch-size-equivalency)
  - [2.4. training on completions only, masking out inputs](#24-training-on-completions-only-masking-out-inputs)
  - [2.5. avoiding Overfitting \& Underfitting](#25-avoiding-overfitting--underfitting)

# 1. how to fine tune gpt-oss

[![how to fine tune gpt%2Doss](https://img.shields.io/badge/how%20to%20fine%20tune%20gpt--oss-link-red)](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)

# 2. LoRA hyperparameters guide

[![LoRA hyperparameters guide](https://img.shields.io/badge/LoRA%20Hyperparameters%20Guide-link-red)](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

## 2.1. what is LoRA?

LoRA is Low-Rank Adaptation. Instead of changing all the model parameters (weights), we instead decompose each weights matrix into a multiplication of thin matrices A and Bight, and optimize those. For instance, this can mean we only optimize 1% of weights.

## 2.2. key fine-tuning hyperparameters

- learning rate | Defines how much the model's weights are adjusted during each training step.
- epochs | The number of times the model sees the full training dataset.
- LoRA or QLoRA | LoRA uses 16-bit precision, while QLoRA is a 4-bit fine-tuning method. LoRA is slightly faster and slightly more accurate, but consumes significantly more VRAM, while QLoRA is slightly slower and slightly less accurate, but uses much less VRAM (4 times less).
- LoRA rank | Controls the number of trainable parameters in the LoRA adapter matrices.
- LoRA alpha | Scales the strength of the fine-tuned adjustments in relation to the rank – $\hat{W} = W + \frac{\text{alpha}}{\text{rank}} \times A B$.
- LoRA dropout | A regularization technique that randomly sets a fraction of LoRA activations to zero during training to prevent overfitting.
- weight decay | A regularization term that penalizes large weights to prevent overfitting and improve generalization.
- warmup steps | Gradually increases the learning rate at the start of training.
- scheduler type | Adjusts the learning rate dynamically during training.
- seed (random state) | A fixed number to ensure reproducibility of results.
- target modules | Specify which parts of the model you want to apply LoRA adapters to — either the attention, the MLP, or both Attention: `q_proj, k_proj, v_proj, o_proj`. MLP: `gate_proj, up_proj, down_proj`.

## 2.3. gradient accumulation and batch size equivalency

- batch size | The number of samples processed in a single forward/backward pass on one GPU.
- gradient accumulation steps | The number of micro-batches to process before performing a single model weight update.
- effective batch size = batch size $\times$ gradient accumulation steps

## 2.4. training on completions only, masking out inputs

The QLoRA paper shows that masking out inputs and training only on completions (outputs or assistant messages) can further increase accuracy by a few percentage points.

## 2.5. avoiding overfitting & underfitting

- overfitting (too specialized)
    - adjust the learning rate | a high learning rate often leads to overfitting, especially during short training runs – for longer training, a lower learning rate may work better
    - reduce the number of training epochs
    - increase weight decay
    - increase lora dropout
    - increase batch size or gradient accumulation steps.
    - dataset expansion | make your dataset larger by combining or concatenating open source datasets with your dataset – choose higher quality ones
    - evaluation early stopping | enable evaluation and stop when the evaluation loss increases for a few steps
    - LoRA alpha scaling | scale the alpha down after training and during inference – this will make the finetune less pronounced
    - weight averaging | literally add the original instruct model and the finetune and divide the weights by 2

- underfitting (too generic)
    - adjust the learning rate | if the current rate is too low, increasing it may speed up convergence, especially for short training runs – for longer runs, try lowering the learning rate instead
    - increase training epochs | train for more epochs, but monitor validation loss to avoid overfitting
    - increase LoRA rank and alpha | rank should at least equal to the alpha number, and rank should be bigger for smaller models/more complex datasets
    - use a more domain-relevant dataset | ensure the training data is high-quality and directly relevant to the target task
    - decrease the effective batch size to 1 | this will cause the model to update more vigorously

