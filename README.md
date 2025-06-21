# From Coding to Cognition: Measuring Reasoning in the Age of o1

## The End of an Era

Two years ago, I built Can-Ai-Code to answer a simple question: could these new language models actually generate syntactically valid code? Back then, we were manually figuring out end-of-sequence tokens, quantization was experimental and potentially destructive, and "which model can code?" was a genuine unknown.

Can-Ai-Code served its purpose. The question has been definitively answered: **yes, AI can code.** But like any good benchmark, success became its downfall. As models improved, everything started clustering at the top. I made harder tests. Models got better. More clustering. The classic benchmark death spiral.

Today, I'm officially retiring Can-Ai-Code. Not because it failed, but because it succeeded so completely that the original question is no longer interesting.

## The New Question: Can AI Think?

With the emergence of reasoning models like o1, we're not asking whether AI can follow patterns anymore—we're asking whether it can *think*. Can it work through novel problems? Can it reason about complex logical structures? Can it maintain working memory across multi-step processes?

But here's the problem: **measuring thinking is fundamentally different from measuring coding ability.**

The same tasks that are trivial for humans can be hillariously difficult for LLMs, how many models still cant count the 'r's in "strawberry", while tasks that seem hard to us might be pattern-matching exercises for them. We needed a completely new approach to evaluation.

## Enter: Parametric Difficulty Scaling

After 10 days of continuous compute on two RTX 3090s (and two blown breakers), I've built something different: a benchmark that **can never be defeated.**

Instead of fixed test cases, I use parametric generators that create unlimited unique problems. Instead of measuring pass/fail, I measure **how far up the difficulty ramp each model can climb.**

The key insight: difficulty has two dimensions:
- **Length**: More elements = more working memory stress  
- **Depth**: More structural complexity = deeper reasoning chains

This creates a 2D difficulty space that we can scale infinitely upward.

## The Architecture of Thought

Testing across 200+ million tokens revealed something remarkable: **different model families have completely different cognitive fingerprints.**

**OpenAI models**: Reasoning virtuosos that crush boolean logic (80%+ accuracy) but struggle with tokenization tasks

**Qwen reasoning models**: Show dramatic improvement from "thinking time" (up to 250% boost for smaller models) 

**Llama**: Balanced generalists with solid baseline performance across all domains

**Phi**: Highly dependent on whether they were architecturally designed for reasoning

These aren't just performance differences—they're different *types of intelligence*.

## Beyond Accuracy: The Three Dimensions of Reasoning

Traditional benchmarks ask: "Did you get it right?"  
The new framework asks three questions:

1. **Height**: How far up the difficulty ramp can you climb?
2. **Efficiency**: How many tokens did you burn getting there?  
3. **Constrained Performance**: How well do you perform with limited resources?

When OpenAI models "defeat" boolean algebra by achieving 80%+ accuracy, that's not the end—it's the beginning. Now we crank up the difficulty and measure efficiency. Can you solve these problems without burning 1000+ reasoning tokens?

## The Benchmark That Evolves

Here's the key innovation: **when models get too good, we automatically make it harder.**

When 2+ models achieve >90% adjusted accuracy, we drop the easiest difficulty bins and add tougher ones. The benchmark literally grows with the field, maintaining discrimination power while tracking the capability frontier.

No more benchmark stagnation. No more clustering at artificial ceilings. Just continuous measurement of an ever-advancing field.

## What We've Learned (So Far)

- **Reasoning capability is architectural, not universal** - it depends on how models were designed and trained
- **"Thinking time" helps, but with diminishing returns** - and the benefits vary dramatically by model family  
- **Working memory is the universal bottleneck** - even SOTA models struggle with bracket stack operations
- **Tokenization remains the Achilles heel** - word sorting defeats nearly everyone
- **There are measurable trade-offs** between reasoning capability and efficiency

## The Road Ahead

This is just the beginning. The current suite focuses on computational reasoning, but the framework extends to any domain where difficulty can be parameterized. Spatial reasoning, causal inference, creative synthesis—all become measurable with the right generators.

I'm also working on visualizing the complete 2D performance surfaces. Nobody has ever seen the actual topology of machine cognition before. What does the landscape of reasoning ability actually look like?  If this is a question you're interested in getting some answers to (and you have some spare compute sitting around), reach out!

## From My Basement to the Future

This research happened on consumer hardware in my basement. Two 3090s, some blown fuses, and a lot of curiosity about whether these systems are actually thinking or just getting very good at statistical pattern matching.

The preliminary results suggest it's both. Different architectures exhibit genuinely different cognitive profiles, but they all hit hard walls when pushed beyond their comfort zones. The question isn't whether AI can think—it's *how* different AIs think, and what the limits of those different approaches are.

Can-Ai-Code asked whether AI could code. That question is answered.

**Can-AI-Think** asks whether AI can reason. That question is just getting started.

---

*The new benchmark suite will be available soon. If you're interested in early access or want to contribute additional reasoning domains, reach out. Fair warning: you might want to upgrade your electrical panel first.*