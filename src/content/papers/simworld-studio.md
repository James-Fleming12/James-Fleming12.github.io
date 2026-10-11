---
title: 'SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning'
description: 'An open-source Unreal Engine 5 platform where a self-evolving coding agent generates embodied learning environments, and environment generation co-evolves with embodied agent learning.'
authors:
    - Haoqiang Kang
    - Xiaokang Ye
    - Yuhan Liu
    - Siddhant Hitesh Mantri
    - Lingjun Mao
    - James Fleming
    - Drishti Regmi
    - Lianhui Qin
pubDate: 5/10/2026
updatedDate: 5/13/2026
venue: 'NeurIPS'
# This paper is hosted on arXiv rather than on this site, so there is no `pdf`
# field (and therefore no embedded viewer). External links are shown as buttons.
links:
    - label: arXiv
      url: https://arxiv.org/abs/2605.09423
    - label: PDF
      url: https://arxiv.org/pdf/2605.09423
    - label: DOI
      url: https://doi.org/10.48550/arXiv.2605.09423
# Canonical URL used in citations; defaults to the paper's page on this site.
url: https://arxiv.org/abs/2605.09423
# `cite: true` shows the citation section. Citations are generated
# automatically, but can be overridden with a manual `citation` block like the
# one below (used here for the arXiv-specific venue).
cite: true
citation:
    text: 'Haoqiang Kang et al. “SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning.” arXiv preprint arXiv:2605.09423, May 2026.'
    bibtex: |
        @misc{kang2026simworld,
          author = {Kang, Haoqiang and Ye, Xiaokang and Liu, Yuhan and Mantri, Siddhant Hitesh and Mao, Lingjun and Fleming, James and Regmi, Drishti and Qin, Lianhui},
          title = {{SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning}},
          year = {2026},
          month = may,
          howpublished = {arXiv preprint arXiv:2605.09423},
          url = {https://arxiv.org/abs/2605.09423}
        }
---

## Abstract

LLM/VLM-based digital agents have advanced rapidly thanks to scalable sandboxes for coding, web navigation, and computer use, which provide rich interactive training grounds. In contrast, embodied agents still lack abundant, diverse, and automatically generated 3D environments for interactive learning. Existing embodied simulators rely on manually crafted scenes or procedural templates, while recent LLM-based 3D generation systems mainly produce static scenes rather than deployable environments with verifiable tasks and standard learning interfaces. We introduce SimWorld Studio, an open-source platform built on Unreal Engine 5 for generating evolving embodied learning environments. At its core is SimCoder, a tool/skill-augmented coding agent that writes and executes engine-level code to construct physically grounded 3D worlds from language/image instructions. SimCoder self-evolves by using verifier feedback (e.g., compilation errors, physics checks, VLM critiques) to revise environments and autonomously add reusable tools and skills to its library. Generated worlds are exported as Gym-style environments for embodied agent learning. SimWorld Studio further enables co-evolution between environment generation and embodied learning: agent performance feedback guides SimCoder to generate adaptive curricula near the learner's capability frontier, so that environments become increasingly challenging as the embodied agent improves. Three case studies on embodied navigation show that self-evolution improves generation reliability, generated environments substantially improve embodied agent performance that generalizes to unseen benchmarks, and co-evolution yields an 18-point success-rate gain over fixed-environment learning and a 40-point gain over an untrained agent.
