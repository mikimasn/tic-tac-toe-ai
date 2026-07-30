# AlphaZero-Based Ultimate Tic-Tac-Toe AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

An artificial intelligence agent for **Ultimate Tic-Tac-Toe**, built from scratch using the AlphaZero architecture (Monte Carlo Tree Search guided by Deep Neural Networks). 

*Note: This repository contains the core algorithmic implementation and training pipeline. It is purely a backend/research project and does not include a GUI or pre-trained weights.*

## Architecture & Implementation

The bot does not rely on hardcoded heuristics. Instead, it learns to play entirely through self-play, using:
*   **Monte Carlo Tree Search (MCTS):** To explore future game states and evaluate moves using the PUCT algorithm.
*   **Deep Neural Networks:** A dual-head PyTorch model containing:
    *   **Policy Head:** Outputs a probability distribution over all possible legal moves.
    *   **Value Head:** Predicts the outcome of the game from the current board state (Win/Loss/Draw) in the range `[-1, 1]`.

## Training Pipeline & Scale

I wrote a custom, automated self-play pipeline to generate training data and optimize the network iteratively.

*   **Training Time:** ~24+ hours on local consumer hardware.
*   **Scale:** Over **20,000 simulated games**.
*   **Structure:** 200 training iterations, with 128 self-play games per iteration.
*   **Loss Functions:** Cross-Entropy Loss (for the policy head) and Mean Squared Error (for the value head).

## Repository Structure

Since this is an algorithmic exploration rather than a consumer app, here is how to navigate the codebase:

*   `Engine.py` - Contains the implementation of the Monte Carlo Tree Search, node expansion logic and the environment logic managing the complex rules and state transitions of Ultimate Tic-Tac-Toe.
*   `Model.py` - PyTorch definitions for the dual-head ResNet/CNN architecture.
*   `train_parallel.py` - The main self-play loop and automated training pipeline.
