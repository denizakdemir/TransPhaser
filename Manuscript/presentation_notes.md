# TransPhaser Presentation Notes

This document contains the speaker notes for the TransPhaser presentation.

## Slide: Introduction: The Phasing Problem
**Presenter Note**: Start by defining the core problem. Explain that standard sequencing gives us the "bag of alleles" for a person but doesn't tell us which ones hang together on the same chromosome. This ambiguity is the "phasing problem."

## Slide: The HLA Challenge
**Presenter Note**: Emphasize that HLA isn't just "another region"—it's the hardest boss in the game. The diversity is massive, meaning "standard" rules of thumb often fail.

## Slide: Deep Dive: Classical EM
**Presenter Note**: Explain the mechanism of EM. Ideally, it's perfect, but computationally, it's a nightmare for long haplotypes. The "E-step" is the killer—it requires checking every possible solution.

## Slide: Deep Dive: Hidden Markov Models (Beagle)
**Presenter Note**: Beagle solves the speed problem by assuming the "Markov Property"—that what happens at locus 10 only depends on locus 9. This makes it fast, but it means it struggles to "remember" long-range patterns, which are common in HLA.

## Slide: Synthesis: How TransPhaser Wins
**Presenter Note**: This is the key conceptual slide. We take EM's rigor but replace the slow E-step with a fast Neural Network. We take Beagle's speed but replace the short-sighted Markov chain with a far-sighted Transformer. It's the best of both worlds.

## Slide: Introducing TransPhaser
**Presenter Note**: This is the hook. TransPhaser isn't just a neural net, and it isn't just statistics. It's both. The "Amortized" part means we learn a function to solve the hard part quickly.

## Slide: TransPhaser vs. The World
**Presenter Note**: Use this table to visually position TransPhaser. The key takeaway is "Native Support" for covariates and "Long-range" dependencies via Transformers.

## Slide: Methods Overview
**Presenter Note**: Transition to the deep dive. Prepare the audience that we are going to look under the hood now.

## Slide: Architecture
**Presenter Note**: Walk through the flow. Data enters the left. The network proposes options. The right side scores them based on biology (HWE, priors).

## Slide: Component 1: Neural Proposal Network
**Presenter Note**: Explain "Amortized". Instead of running a fresh optimization for every person (like standard EM), we train a network once to just "know" the answer. It narrows the search space from millions to just 64 candidates.

## Slide: Proposal Network Details
**Presenter Note**: Highlight the Transformer choice. It's not just a buzzword; it's essential for capturing the complex LD patterns in HLA that don't just decay with distance.

## Slide: Component 2: Conditional Haplotype Prior
**Presenter Note**: This is an "Smoothed" frequency table. Instead of a giant lookup table (which fails for rare alleles), we learn a function that tells us how likely a sequence is, allowing generalization.

## Slide: Component 3: HWE Prior
**Presenter Note**: This grounds the model in biology. We assume the parents were independent (mostly). This is a strong constraint that guides the neural network.

## Slide: Component 4: Constrained Emission Model
**Presenter Note**: Crucial point. Neural nets hallucinate. This logic gate ensures we never output a haplotype that contradicts the observed data.

## Slide: The Training Loop (Neural EM)
**Presenter Note**: Explain the loop. It's a self-reinforcing cycle. The network proposes, biology scores, and the network learns from that score to propose better next time.

## Slide: Synthetic Data Generation
**Presenter Note**: Address the data source honestly. We used synthetic data to have a perfect "answer key" to grade against. We made it hard on purpose (40% rare/recombinant).

## Slide: Experiment 1: Multi-Population
**Presenter Note**: Set the stage for the first win. This is the "ideal world" scenario where we know where people are from.

## Slide: Results: Multi-Population
**Presenter Note**: Point out the margin. 6% in this field effectively closes the gap significantly. 85% is very high for 6-locus HLA.

## Slide: Experiment 2: Single-Population (Hard Mode)
**Presenter Note**: This is the "stress test". Can it win on a level playing field without extra info?

## Slide: Results: Single-Population
**Presenter Note**: The gap is smaller, but it's still there. This proves the core engine is robust.

## Slide: Experiment 3: Frequency Prediction
**Presenter Note**: Shift gears. Sometimes we don't care about the patient; we care about the population stats (epidemiology).

## Slide: Results: Frequency Prediction
**Presenter Note**: A tie is a win here. It means we don't lose population accuracy to gain individual accuracy. We get both.

## Slide: Discussion: Why it works
**Presenter Note**: Synthesize the technical "why". It's about data representation (vectors vs symbols) and memory (attention vs Markov).

## Slide: Limitations
**Presenter Note**: Be humble. Admit the synthetic data constraint. It's standard for method development but requires real-world follow-up.

## Slide: Conclusion
**Presenter Note**: Summarize the main number and the main concept.

## Slide: Thank You
**Presenter Note**: Open the floor.
