# Presentation Text for PPP Pipeline

## Stage 1: Data Collection

In this first stage, we extract three types of information from 241,186 words of text. We collect baseline linguistic features like word length, frequency, and position in the sentence. At the same time, we pass this text through our language model to compute surprisal values, which tell us how unexpected each word is to the model. Finally, we have human participants read this same text while we track their eye movements to measure reading times in milliseconds. This parallel extraction allows us to test whether the model's uncertainty about words matches human reading difficulty.

## Stage 2: Model Fitting

We build two statistical models to predict human reading times. Model A is our baseline, predicting reading time using only traditional linguistic features like word length and frequency. Model B is identical to Model A but adds the model's surprisal values as additional predictors. Both models use Maximum Likelihood Estimation to find the optimal parameters. After fitting, Model A achieves a log-likelihood of -1,435,663 with 12 parameters, while Model B achieves -1,433,935 with 15 parameters. The goal is to test whether adding surprisal meaningfully improves predictions beyond what traditional features already capture.

## Stage 3: Comparison and Results

From Stage 2, we have two log-likelihoods: Model A at -1,435,663 and Model B at -1,433,935. The difference is 1,727.61, showing Model B fits better. We run two tests on this difference. First, PPP divides by our sample size giving 0.007163 nats per word. Second, the likelihood ratio test multiplies by 2, yielding LR = 3,455 with p < 0.0001. This proves Model B significantly outperforms Model A. Our hypothesis is confirmed: when the model finds a word surprising, humans slow down. This validates human-like language processing.

## Research Question

Do language models process text like humans? We approach this by comparing model surprisal with human reading times. This research bridges AI and cognitive science by validating that language models can serve as cognitive tools for understanding human language processing.

## Summary

We tested whether language models think like humans by comparing model predictions with reading times from 241,186 words. The model that included surprisal significantly outperformed the baseline model with p < 0.0001 and PPP = 0.007 nats per word. Our conclusion is clear: these models capture human-like language processing patterns.
