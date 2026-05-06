---
title: Generative AI from Variational Inference
description: Variational Inference, VAEs,  Diffusion Models
pubDate: 5/04/2026
---

## Problem Setup:
We already know that function approximation works really well for tasks like classification and prediction, but let's consider the task of generation. To show why this is meaningfully different, we can talk about the nature of both types of tasks. Tasks like prediction have a convergent nature, where we take a massive and chaotic input and compress it down to a single deterministic output. For example in the task of classifying cats or dogs, we take the million or so pixels in the image and return out a deterministic output for each image. In contrast, generation tasks have a divergent nature, flipping the complexity of both sides of the task over. Instead, we are given a very simple input of "Generate a Cat" and need to output the million or so pixels that are required in the image.

This ...

## Variational Inference:

## Mean-Field Approximation:

## Variational Autoencoders:

## Normalizing Flows:

## Diffusion Models:
