
## miniKAF_box: A Mini Toolbox Kernel Adaptive Filtering and Neural Networks for GNU Octave

**miniKAF_box** is a lightweight, pedagogical, and functional toolbox for Kernel Adaptive Filtering (KAF) and Neural Networks. It is designed specifically for researchers, students, and engineers who need a transparent and easy-to-debug implementation of state-of-the-art algorithms in GNU Octave.

## Why use miniKAF_box?

While inspired by the well-known [KAFBOX by Steven Van Vaerenbergh](https://github.com/stevenvw/kafbox), this version offers a simplified architecture, making it easier to integrate into research projects and academic benchmarks

## Algorithms Included

This toolbox contains simplified implementations of the following models:

- **KRLS** (Kernel Recursive Least Squares)
- **KRLS-T** (Kernel Recursive Least Squares Tracker)
- **SKRLS** (Sparse Kernel Recursive Least Squares)
- **KNLMS** (Kernel Normalized Least Mean Squares)
- **QKLMS** (Quantized Kernel Least Mean Squares)
- **SVR** (Support Vector Regression - optimized function for LIBSVM, a Octave library)
- **MLP** (Multi Layer Perceptron)
- **LSTM** (A custom, native GNU Octave implementation of Long Short-Term Memory networks)

## Research & Benchmarking

This repository is being developed as a baseline for comparative research between Kernel methods and Deep Learning models. It provides a common framework to compare traditional KAF algorithms against custom-built LSTM and MLP architectures in non-linear regression tasks.

### Reference
If you find this toolbox useful for your research, please link back to this repository.
*(Formal citation for the upcoming paper will be added here soon).*

## How to Install
1. Clone this repository:
   ```bash
   git clone https://github.com/kelvinsales/miniKAF_box.git

2. Add the folders to your Octave path:
    
    addpath(genpath('miniKAF_box'));
    savepath();

## Requirements

  GNU Octave (4.0 or higher recommended)

Developed by Allan Kelvin M. Sales.
Simplified implementations based on the original concepts of Steven Van Vaerenbergh's kafbox.
