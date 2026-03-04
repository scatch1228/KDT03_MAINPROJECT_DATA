# 1. Baseline FlowPredictor (Stacked LSTM)
### Multi-Layer Tapered Recurrent Neural Network

The **FlowPredictor** is the baseline model for the reservoir project. It utilizes a deep stacked LSTM architecture to extract hierarchical temporal features from 180 minutes of multivariate history to predict a 15-minute future window in a single forward pass.

## Architecture Overview
The model follows a "tapered" design where each subsequent layer reduces the feature space, forcing the model to learn the most compressed and salient representations of the flow data.

* **Layer 1 (Input):** A standard LSTM processing `input_dim` (4) into `hidden_dim`.
* **Layer 2 (Compression):** An LSTM that reduces the hidden state by half (`hidden_dim // 2`).
* **Layer 3 (Bottleneck):** A final LSTM layer that reduces the representation again (`hidden_dim // 4`).
* **Dropout:** Applied after every layer to prevent overfitting during the training of the 180-minute sequences.
* **Head:** A single Linear (Fully Connected) layer that maps the **last hidden state** of the 3rd LSTM directly to the `output_dim` (15 minutes).



## Key Characteristics
* **Hierarchical Feature Extraction:** By stacking LSTMs, the model can capture different "scales" of time (e.g., Layer 1 captures minute-to-minute noise, while Layer 3 captures the 3-hour trend).
* **Many-to-One Mapping:** While the output is 15 minutes, the model uses the **final time-step's hidden state** (`out[:, -1, :]`) to make the prediction. This assumes the final state has successfully "memorized" the entire 180-minute window.
* **Tapered Design:** The reduction in dimensions (`128 -> 64 -> 32`) acts as a regularizer, preventing the model from simply memorizing the training data.

## Dimensionality Flow
1.  **Input:** `[Batch, 180, 4]`
2.  **LSTM 1:** `[Batch, 180, 128]`
3.  **LSTM 2:** `[Batch, 180, 64]`
4.  **LSTM 3:** `[Batch, 180, 32]`
5.  **Final State Slice:** `[Batch, 32]` 
6.  **Output (FC):** `[Batch, 15]`

=================================================

# 2. FlowTransformer (Encoder-Only Architecture)
### Multi-Head Self-Attention for Time-Series Forecasting

The **FlowTransformer** leverages the power of Self-Attention to identify non-linear relationships across a 180-minute window of reservoir data. Unlike recurrent models, it processes the entire history in parallel to extract global temporal features.

## Architecture Overview
The model is an **Encoder-only Transformer** designed for high-throughput temporal feature extraction.

* **Input Projection:** A linear layer that expands the `input_dim` (9) into the model's internal representation space (`d_model=64`).
* **Sinusoidal Positional Encoding:** Since Transformers have no inherent sense of order, this layer injects a "time signature" into the data using fixed sine and cosine functions. 

* **Stacked Transformer Blocks:** 3 layers of Multi-Head Self-Attention (`n_head=8`). Each head focuses on different aspects of the history (e.g., one head might look for rainfall spikes, while another tracks gradual temperature shifts).
* **Feed-Forward Expansion:** Each block uses a latent expansion of `d_model * 4` (256) to process complex patterns.
* **Global Summary (Last-Step):** The model extracts the representation of the **final time step** from the transformer output to represent the entire 3-hour history.
* **Linear Head:** Maps the final 64-dimensional latent vector directly to the 15-minute forecast.

## Key Characteristics
* **Parallel Attention:** Every minute in the 180-minute history can directly attend to every other minute, regardless of how far apart they are.

* **Encoder-Only Design:** By focusing on the Encoder, the model excels at "understanding" the past history before making a one-shot prediction for the future.
* **Dimensionality:** * Input: `[Batch, 180, 9]`
  * Projected & Permuted: `[180, Batch, 64]` (Standard PyTorch Transformer format)
  * Output: `[Batch, 15]`

## Technical Specifications
* **Heads:** 8 (Parallel attention streams)
* **Layers:** 3 (Depth of the transformer stack)
* **Embedding Size ($d_{model}$):** 64
* **Dropout:** 0.1 (Used within attention and feed-forward blocks)

=================================================

# 3. Seq2Seq LSTM with Additive Attention
### Recurrent Encoder-Decoder with Dynamic Contextual Search

The **LSTMSeq2SeqAttnModel** is the most advanced architecture in this project. It transitions from "One-Shot" mapping to an **Iterative Generation** strategy, allowing the model to focus on different historical events for each specific minute of the 15-minute forecast.
This model is still in experiment and not included in the service. The source code for this model is in the notebook directory.

## Architecture Overview
This model consists of two distinct LSTM networks that communicate via a shared "Context" and an Attention mechanism.

* **The Encoder (History Processor):** A multi-layer LSTM that reads the 180-minute window. It creates a "Memory Bank" (hidden states) of every single minute of the past.
* **Additive Attention (The Search Engine):** For every minute of the forecast, the model calculates a set of weights ($e_t$) using a learned alignment function:
  $$score(s_t, h_i) = V^T \tanh(W_1 s_t + W_2 h_i)$$
  This allows the model to "look back" at the 180-minute history and pick exactly which minutes matter right now.


* **Step Embeddings (The Clock):** To prevent "Time Blindness," the decoder receives a unique vector (Embedding) for each of the 15 forecast steps. This tells the model exactly where it is in the 1-to-15 minute timeline.
* **The Decoder (The Generator):** Uses an `LSTMCell` to predict the flow minute-by-minute. At each step, it combines the **previous prediction**, the **current step embedding**, and the **attention context**.

## Key Characteristics
* **No Information Bottleneck:** Unlike the Baseline LSTM or Transformer, this model does not rely on a single "final state." It can access all 180 minutes of history at any time via `torch.bmm`.
* **Autoregressive Feedback:** Each predicted minute informs the next, ensuring the 15-minute curve maintains a physically realistic shape.
* **Temporal Awareness:** The `step_embedding` ensures the model distinguishes between the immediate future (Minute 1) and the edge of the horizon (Minute 15).


## Dimensionality Flow
1. **Encoding:** Input `[Batch, 180, 9]` $\to$ Memory Bank `[Batch, 180, 128]`
2. **Attention:** Query ($h_{dec}$) + Memory $\to$ Context Vector `[Batch, 1, 128]`
3. **Decoding Step:** [Context + Prediction + Step_Embed] $\to$ `LSTMCell`
4. **Output Head:** `[Batch, 128]` $\to$ `[Batch, 1]` (Repeated 15 times)

## Technical Specifications
* **Hidden Dimension:** 128
* **Attention Type:** Additive (Bahdanau)
* **Embedding Dimension:** 16
* **Forecast

=================================================

## Architecture Comparison Matrix

| Feature | **Traditional LSTM** (Baseline) | **FlowTransformer** (Encoder-Only) | **Seq2Seq + Attention** (Advanced) |
| :--- | :--- | :--- | :--- |
| **Core Engine** | 3-Layer Tapered LSTM | Multi-Head Self-Attention | LSTM Encoder + LSTMCell Decoder |
| **History Processing** | Sequential (One-by-one) | Parallel (All-at-once) | Sequential + Dynamic Re-scanning |
| **The "Bottleneck"** | **Final Hidden State** (Layer 3) | **Final Time-Step** (Encoder) | **None** (Attends to all 180 mins) |
| **Time Sensing** | Implicit (Order of recurrence) | **Positional Encoding** (Sin/Cos) | **Step Embeddings** (Minute 1-15) |
| **Mapping Strategy** | One-Shot (128 $\to$ 15) | One-Shot (64 $\to$ 15) | **Iterative** (1 min at a time) |
| **Best For** | Stable, smooth flow trends | Complex global correlations | High-precision, dynamic curves |
