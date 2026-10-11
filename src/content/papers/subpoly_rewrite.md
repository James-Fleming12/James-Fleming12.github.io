---
title: 'Subpolynomial query complexity for well-conditioned log-concave sampling: An expository account of the proof, with consequences for diffusion models'
description: 'An exposition of the subpolynomial query-complexity proof for well-conditioned log-concave sampling, and its consequences for diffusion models.'
authors: ['James Fleming']
pubDate: 10/11/2026
venue: 'Preprint'
pdf: '/papers/subpoly_rewrite.pdf'
# This is an expository AI rewrite, so no citation section is shown.
cite: false
# Show the AI disclaimer before the abstract.
ai: true
---

## Abstract

In September 2026 OpenAI posted a manuscript, written by an internal model, proving that the optimal dimension exponent for sampling well-conditioned log-concave densities in the exact first-order oracle model is zero. For every fixed $\varepsilon > 0$ the manuscript's algorithm uses at most $C_\varepsilon d^\varepsilon$ evaluations of the potential's gradient on *every* execution, for $C^2$ potentials on $\mathbb{R}^d$ with a known minimizer and Hessian between $I_d$ and $2I_d$. It also proves an $\Omega(\log d)$ lower bound for arbitrary randomized adaptive algorithms.

The manuscript is complete but compressed. The machinery is developed in the order the proof needs it, and the reader is rarely told *why* a step is there. This note has two purposes. The first is expository, inspired by Lance Fortnow's rewrites of UGC and L=RL=BPL. It presents the same proof for a reader who knows the elements of log-concave sampling (Langevin discretization, functional inequalities, Wasserstein comparison) but has not read the manuscript. It explains what each step is for, carries out the central calculations in full (the all-split tensor estimates, the material-derivative bookkeeping, the stationary centering flow, the noisy finite evaluation of nested conditional means, and the Fourier-type lower-bound argument), and sketches the parts that adapt known machinery, with pointers to the original.

The second purpose is to draw out what the result means for diffusion models, where the score is normally learned from samples. The denoising score and denoiser of the smoothed target are exactly the conditional mean the construction evaluates, every noise level can be accessed with subpolynomially many first-order queries, and the manuscript's sampler is an ancestral denoiser whose convergence theorem is the upper bound. It is an exposition, not a referee report.
