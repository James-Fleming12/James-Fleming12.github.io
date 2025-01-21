---
title: Building Up to SAMURAI
description: Covering SAM, SAM 2, and SAMURAI
pubDate: 1/21/25
---
Segment Anything Models (often shortened to SAM) represent a model of promptable image and video segmentation that allow Zero-Shot Learning capabilities, meaning they can perform on data they weren't trained on and have never seen before. They are the new tier of Vision Transformer models and are initiating an age of the most flexible Computer Vision that has ever been. [SAMURAI](https://arxiv.org/abs/2411.11922) represents one of the newest models that helps to segment long videos where old models had problems with object movement and occlusion. This post will cover SAM, SAM 2 (it's extension to videos), and SAMURAI (which adds more memory features), but will skip over the training procedures of each as well as the data engine for how the original papers generated their datasets. This also assumes you know general Transformer structure as well as typical addons in between layers, although knowing them is not necessary to get the general points.

## Segment Anything Model (SAM):
The original SAM model presents the general architecture for promptable image segmentation. It follows a general Encoder-Decoder structure, with two Encoders and a single Decoder that outputs the segmentation, with the below specifications. The Encoders represent the input in a lower dimensional structure and the Decoder alters them in an Attention structure to generate the final outputs.

1. Image Encoder (produces Image Embeddings)
2. Prompt Encoder (produces Prompt Embeddings)
3. Mask Decoder (processes both into a segmentation mask)

### Encoders:
The Image Encoder has a simple architecture, with it being a [Masked Autoencoder](https://arxiv.org/abs/2111.06377)-based [Vision Transformer](https://arxiv.org/abs/2010.11929) (some alterations during training to help it learn a better representation of its inputs). This creates a lower resolution embedding that is downscaled $16\times$ from the original image.

Prompts are any sort of information that can be given to help the segmentation process. This includes points, boxes, text inputs, and even other masks. These different kinds of prompt inputs are then used to separate prompts into both Sparse (points, boxes, text) and Dense (masks) prompts, wich each prompt type having a different mindset when it comes to encoding (sparse prompts are turned into tokens, dense prompts added to the image embedding). The method of embedding is shown below for each type of prompt.

1. Point (sum of a pair of positional encodings, one for the point's location and one of two learned embeddings for whether it's in the background or foreground)
2. Box (an embedding pair of the position of both the top-left and bottom-right corner)
3. Text (processed by [CLIP](https://arxiv.org/abs/2103.00020) image encoder)
4. Mask (processed by a small CNN with GELU and LayerNorms, and then added to the Image Embedding element-wise)

### Mask Decoder:
Once all the embeddings are generated, they need to be processed. This is done through a decoder structure followed by a couple of post processing steps. The decoder receives as input the image embedding and all prompt embeddings along with an output token, a learned embedding which is used later to enhance the mask generated. The general decoder structure is described below (with all the embeddings excluding the image embedding being simple called "tokens"). In between each layer described their are typical Residual Connections, LayerNorms, and a [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) mechanism. As well the information in each is refreshed between each layer. Positional encodings are added to the image embedding in between each step and the original prompt tokens are re-added to the updated tokens at each step.

1. Self-Attention on the tokens (tokens update eachother)
2. Cross-Attention from the Token to the Image Embedding (Image Embedding updates each Token)
3. Point-Wise MLP for each Token (updates tokens with learned parameters)
4. Cross Attention from the Image Embedding to the Tokens (Tokens update Image Embedding)

After the decoder is finished processing the Image Embedding and Output Token receive a couple more processing steps. The Image Embedding is upscaled with some Transposed Convolutional Layers (a form of Deconvolutional Layer that can also morph the input). The Output Token is passed through a 3 layer MLP (after being updated with Cross Attention and the Image Embedding one more time) to match the dimensions of the Image Embedding. These are then combined with a spatially point-wise product between the Image Embedding and Output Token (which is done by flattening the Image Embedding and then unflattening it after the product).

### Ambiguous Prompts:
To help the flexibility of the model even more it has built in systems to help with ambiguous prompts. A prompt is considered ambiguous if it only consists of one prompt type which can then correlate with a large number of valid masks. In these cases, the Decoder receives three Output Tokens instead of one, which outputs three masks each with their own confidence score. Each is then combined using another small head with it's own output token that tries to calculate the Intersection over Union (IoU) of all three. During training, all three are still generated but only the lowest loss between the three of them is propagated back through the network.

## Segment Anything Model 2 (SAM 2):
SAM 2 extends the original SAM model by extending it's inputs to videos and improving its performance by adding some features that work with the memory of the model. An additional Memory Attention step is added after the Image Encoder to process the Image Embedding before the Mask Decoder, which incorporates information from an improved Memory Bank of past frames which are used to process a sequence of frames with the same prompts. It also implements a [Hiera](https://arxiv.org/abs/2306.00989) model instead of the ViT for the Image Encoder which are used for skip connections to the Decoder as well as using both Sinusoidal Positional Embeddings and [Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864).

### Memory Attention:
The Memory Attention follows a similar structure the decoder block described above. It works to incorporate information from past frames features from the Memory Bank on the current frames. It uses traditional attention to benefit from the advancements in efficiency made in [FlashAttention-2](https://arxiv.org/abs/2307.08691) with a structure defined below.

1. Self Attention on the current Image Embedding
2. Cross Attention to memories of past frames and object pointers (described later)
3. An MLP

### Memory Bank:
The Memory Bank stores two key pieces of information from past frames, Spatial Feature Maps (image embeddings fused with the predicted mask) and Object Pointers (output tokens). These also recieve additional temporal positional encoding. The first processed frame and the most recent $N$ frames are held and used for the Memory Attention.

## SAMURAI:
SAMURAI (which stands for SAM-based Unified and Robust zero-shot visual tracker with motion Aware Instance-level memory) improves on SAM 2 with a methodology of movement prediction. It keeps the general architecture of SAM 2 but adds a post processing Motion Modeling segment after the Mask is output as well as improved Memory Selection.

### Motion Modeling:
The model uses a [Kalman Filter](https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf) (a method of predicting the state of a dynamic system) to try and map the motion of the objects that need to be mapped. This system chooses the most confident mask out of $N$ candidates that are output by the model. I have absolutely no clue how Kalman filters work and I do not intend to learn absolutely any control theory any time soon, but the general procedure is very understandable without an in-depth understanding of the field if the gaps in between aren't necessary to know. First, the state of each object is modeled using the following with some position $(x,y)$ and dimensions $(w,h)$ with $\dot{n}$ being the velocity of some variable $n$.
$$
x=[x,y,w,h,\dot x,\dot y,\dot w,\dot h]^T
$$
This is then used to generate a state prediction $\hat{x}_{t+1|t}$ (the prediction for the next timestep given the information at the current timestep) for each mask $\mathcal{M}_i$. Some linear state transition matrix $F$ (describes how the system evolves from one timestamp to the next) is used along with the previous state. This then generates some IoU score $s_{kf}$ (which will be very important later) which is then used to choose the best fitting mask $\mathcal{M}^*$ out of the set of masks with some Hyperparameter $\alpha$ to control how the two are mixed (with $s_{mask}$ being described later).
$$
\begin{gather*}
\hat{x}_{t+1|t}=F\hat{x}_{t|t}\\
s_{kf}=\text{IoU}(\hat{x}_{t+1|t,\mathcal{M}})\\
\mathcal{M}^*=\text{arg max}_{\mathcal{M}_i}(\alpha_{kf}\cdot s_{kf}(\mathcal{M}_i)+(1-\alpha_{kf})\cdot s_\text{mask}(\mathcal{M}_i))
\end{gather*}
$$
This optimized mask is then used as the bounds for a bounding box $z_t$ whose dimensions change how the update is performed. $K_n$ is the Kalman gain (adjusts how much a prediction is corrected) and an observation matrix $H$ (translates back into the given measurement's space).
$$
\hat{x}_{t|t}=\hat{x}_{t-1|t}+K_t(z_t-H\hat{x}_{t|t-1})
$$

### Memory Selection:
In order to accurately select the frames with the most data available, the model uses a system of three scores per frame. An affinity score $s_{mask}$ is generated for each mask with an Affinity Head after the Decoder, an object score $s_{obj}$ is generated for the entire frame from an Object Head after the Decoder, and the motion score $s_{kf}$ is derived from above. If the values of these three match some preset thresholds ($\tau_{mask}$, $\tau_{obj}$, and $\tau_{kf}$ respectively), then they are considered for the memory bank $B_t$ as long as they are within the most recent $N_{max}$ frames considered.
$$
B_t=\{m_i|f(s_{mask},s_{obj},s_{kf})=1,t-N_{max}\leq i\leq t\}
$$