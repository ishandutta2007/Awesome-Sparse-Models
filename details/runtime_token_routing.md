# Runtime Token-Routing (Mixture of Experts)

## Overview
Runtime token routing dynamically routes each incoming token to a subset of specialized experts (sub-networks) within the model, keeping the active parameter count per token constant while scaling total parameters.

## Architecture & Flow
Below is a diagram representing the mechanics of **Runtime Token-Routing (Mixture of Experts)**:

```mermaid
graph TD
    A[Input Token x] --> B(Gating/Router Network)
    B --> C{Top-K Expert Selection}
    C -->|Select Expert 1| D[Expert 1 FeedForward]
    C -->|Select Expert 2| E[Expert 2 FeedForward]
    C -->|Skip| F[Expert N FeedForward]
    D --> G(Weighted Summation)
    E --> G
    G --> H[Output Token Layer]
```

## Further Details
This component is vital to the implementation and optimization of modern sparse deep learning systems. It helps scale the parameter capacity of neural architectures while maintaining efficiency at training and inference time.

---
[← Back to README](../README.md)
