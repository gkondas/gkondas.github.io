---
title: "EveryQuery: A Promptable Foundation Model for Clinical Prediction Tasks over Electronic Health Records"
collection: publications
permalink: /publication/2026-03-09-everyquery
excerpt: 'EveryQuery takes a structured clinical query and a patient history as input and estimates the outcome directly in a single forward pass, instead of simulating many patient trajectories.'
date: 2026-03-09
venue: 'Structured Data for Health Workshop, ICML 2026'
paperurl: 'https://arxiv.org/abs/2603.07900'
citation: 'Payal Chandak, <b>Gregory Kondas</b>, Liat Antwarg Friedman, Isaac Kohane, Matthew McDermott. (2026). &quot;EveryQuery: A Promptable Foundation Model for Clinical Prediction Tasks over Electronic Health Records.&quot; <i>Structured Data for Health Workshop, ICML 2026</i>.'
---

Autoregressive EHR foundation models must simulate many patient trajectories at inference time, which is expensive and noisy for rare events. EveryQuery instead takes a structured clinical query as a prompt alongside the patient's history and estimates the outcome directly. It outperforms a competitive autoregressive baseline on classification and time-to-event tasks across three EHR datasets, with the largest gains on rare outcomes and far less inference compute.

[Paper](https://arxiv.org/abs/2603.07900)
