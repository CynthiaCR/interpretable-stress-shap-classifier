# SHAP vs. Coefficients: Does SHAP Agree With What the Model Actually Learned?

**Status:** In-Progress

## Problem Statement

LLMs and other complex models are widely known as "black boxes", and interpretability tools like SHAP are increasingly used to explain their predictions. But SHAP's explanations are only trustworthy if they actually reflect what a model is doing. That claim is hard to verify directly on a black-box model since there's no ground truth to check against.

This project navigates around that problem by using **logistic regression** as a controlled setting where ground truth *does* exist: its coefficients are exact and interpretable by construction, so SHAP's output can be checked against them directly.

## Hypothesis

**H1** (exploratory, not a formal statistical test): SHAP's feature rankings will largely agree with the logistic regression's raw coefficient rankings, but will diverge for features that are rare in the dataset or interact with other features, revealing cases where a coefficient alone would be misleading.

## Data & Model

- Dataset: Dreaddit (Reddit posts labeled for stress).
- Model: Logistic Regression with TF-IDF features.

## Methodology

1. Preprocess Dreaddit and train a Logistic Regression + TF-IDF baseline.
2. Extract the top 20 features by raw coefficient magnitude.
3. Apply `shap.LinearExplainer` to compute SHAP values for the trained model.
4. Extract the top 20 features by **mean absolute SHAP value**.
5. Compare the SHAP top-20 and coefficient top-20 lists; compute the overlap between the two.
6. For features that appear in one top-20 list but not the other:
   - Check document frequency (rare vs. common)
   - Compute pairwise correlations with other features to identify likely explanations for the divergence.
7. Generate a global summary plot and 3-5 individual force/waterfall plots.

## Success Criteria

Overlap between the two top-20 lists is the primary measure of agreement. There is no pass/fail threshold. The degree and pattern of agreement/divergence, and the reasons behind it, are the finding.

## Scope Boundaries (what this is NOT)

- Not a statistical hypothesis test. "Agreement" is assessed by overlap count, not a formal statistical measure.
- Not a claim about model reliability or readiness for real-world/clinical use.
