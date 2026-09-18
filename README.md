# Deep Learning PR 2 - Adult Income Dataset (ANN Binary Classification)

## Project Title
Deep Learning PR 2: Predicting Income Level using an Artificial Neural Network

## Dataset Name
Adult Income Dataset (also known as the Census Income dataset)

## Dataset Source
UCI Machine Learning Repository

## Dataset URL
https://archive.ics.uci.edu/dataset/2/adult

## Project Objective
The goal of this project is to build an ANN / MLP binary classification model that predicts
whether a person's annual income is `<=50K` or `>50K`, using demographic and work related
features such as age, education, occupation, hours worked per week, and marital status.
Along the way, the project studies how different deep learning choices (activation functions,
weight initialization, loss functions, batch normalization, and optimizers) affect model
performance on this dataset.

## Preprocessing Steps
1. Stripped whitespace from all text (object) columns.
2. Replaced `'?'` values with `NaN` and dropped rows with missing values (found in `workclass`,
   `occupation`, and `native.country`).
3. Dropped the `fnlwgt` column (a census sampling weight, unrelated to income) and the
   `education` column (redundant with `education.num`).
4. Encoded the target column `income` as 0 (`<=50K`) and 1 (`>50K`).
5. One hot encoded all categorical columns using `pd.get_dummies(drop_first=True)`.
6. Scaled the numeric columns (`age`, `education.num`, `capital.gain`, `capital.loss`,
   `hours.per.week`) using `StandardScaler`. The one hot encoded binary columns were left
   unscaled.
7. Split the data into training and test sets (80/20) using `train_test_split` with
   `random_state=42` and `stratify=y` to preserve the class balance.

## Explanation of the Six Main Concepts

**1. Activation Functions** - ReLU, Tanh, Sigmoid, and ELU were compared as hidden layer
activations. ReLU is the most common choice but can suffer from "dead neurons" (neurons that
always output 0). ELU avoids this by allowing small negative outputs. Sigmoid in hidden layers
can cause vanishing gradients since its derivative is at most 0.25.

**2. Weight Initialization** - Zero initialization causes a symmetry problem where every neuron
in a layer learns the same thing, so the network cannot learn properly. Glorot (Xavier)
initialization is designed for sigmoid/tanh activations, while He initialization is designed for
ReLU based networks, since it accounts for the fact that ReLU zeroes out roughly half of its
inputs.

**3. Loss Functions** - Binary Cross Entropy (BCE) is the standard loss for binary classification.
Since the dataset is imbalanced, Weighted BCE (using `compute_class_weight`) and Focal Loss (which
down-weights easy examples and focuses on hard ones) were also tested to see their effect on
minority class performance, compared against plain BCE and MSE.

**4. Batch Normalization** - Normalizes the output of a Dense layer before the activation
function, using batch statistics during training and a moving average during inference. It has
learnable gamma (scale) and beta (shift) parameters.

**5. Optimizers** - SGD, SGD with Momentum, RMSprop, Adam, and an explicitly configured Adam were
compared. Adam combines momentum (first moment) with an adaptive learning rate based on squared
gradients (second moment), using bias correction on both moving averages.

**6. Class Imbalance Handling** - The dataset is around 75% class 0 and 25% class 1. Because of
this, accuracy alone can be misleading, so Precision, Recall, F1-score (for class 1), and ROC-AUC
were tracked throughout the notebook, along with techniques like class weighting to help the model
better detect the minority class.

## build_ann() Parameter Table

| Parameter | Description | Default |
|---|---|---|
| `input_dim` | Number of input features | required |
| `hidden_units` | List of neuron counts for each hidden layer | `[128, 64]` |
| `activation` | Activation function used in hidden layers | `'relu'` |
| `initializer` | Kernel weight initializer | `'glorot_uniform'` |
| `use_batch_norm` | Whether to add BatchNormalization after each Dense layer | `False` |
| `optimizer` | Optimizer used to compile the model | `'adam'` |
| `loss` | Loss function used to compile the model | `'binary_crossentropy'` |

## Important Plots
All plots are saved in the `plots/` folder, including:
- `eda_class_balance.png`, `eda_age_distribution.png`, `eda_hours_boxplot.png`,
  `eda_education_countplot.png`, `eda_correlation_heatmap.png`
- `baseline_training_curves.png`, `baseline_confusion_matrix.png`
- `activation_comparison.png`, `relu_activation_distribution.png`, `sigmoid_gradient_flow.png`
- `initialiser_convergence.png`, `zero_init_failure.png`, `weight_distributions.png`
- `bce_vs_mse.png`
- `batchnorm_dynamics.png`, `batchnorm_position.png`, `batchnorm_gamma_beta.png`
- `optimiser_convergence.png`, `learning_rate_sensitivity.png`
- `roc_curves.png`, `results_table.png`

## Final Results Table (from the actual notebook run)

| Model | Activation | Init | Loss | BatchNorm | Optimizer | Test Acc | Prec(1) | Recall(1) | F1(1) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ReLU | Glorot Uniform | BCE | No | Adam | 0.8415 | 0.7262 | 0.5785 | 0.6440 | 0.8916 |
| Best Activation (ELU) | ELU | Glorot Uniform | BCE | No | Adam | 0.8494 | - | - | 0.6729 | - |
| Best Initializer (He Normal) | ReLU | He Normal | BCE | No | Adam | 0.8373 | 0.6839 | 0.6387 | 0.6605 | 0.8893 |
| Weighted BCE | ReLU | He Normal | Weighted BCE | No | Adam | 0.8044 | 0.5723 | 0.8345 | **0.6790** | 0.8917 |
| Focal Loss | ReLU | He Normal | Focal Loss | No | Adam | 0.8262 | 0.7815 | 0.4148 | 0.5420 | 0.8911 |
| BatchNorm | ReLU | He Normal | BCE | Yes | Adam | 0.8389 | 0.6993 | 0.6142 | 0.6540 | **0.8959** |
| Final Combined | ReLU | He Normal | BCE | Yes | Adam | 0.8439 | 0.7234 | 0.5995 | 0.6556 | 0.8954 |

Based on this run, **Weighted BCE** produced the highest class-1 F1-score (0.6790), because it
directly boosts recall for the minority class by making the model pay a bigger loss penalty for
missing a `>50K` case. **BatchNorm** produced the highest ROC-AUC (0.8959). These are the results
observed in this specific run, not universal guarantees.

## Video Section
See `video_script.txt` for a 5-10 minute spoken explanation script covering the dataset, all
preprocessing steps, and every experiment in the notebook, based on the actual results above.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow / Keras, Jupyter Notebook

## Repository Structure
```
DL_PR2/
│
├── DL_PR2.ipynb
├── DL_PR2.html
├── README.md
├── requirements.txt
│
└── plots/
    ├── eda_class_balance.png
    ├── activation_comparison.png
    ├── initialiser_convergence.png
    ├── weight_distributions.png
    ├── bce_vs_mse.png
    ├── batchnorm_dynamics.png
    ├── optimiser_convergence.png
    ├── roc_curves.png
    └── results_table.png
```
