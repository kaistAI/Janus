---
layout: default
---

## Abstract

- Current large language model (LLM) alignment methods assume that aligning with general public preferences is optimal, but this overlooks the diverse and nuanced preferences individuals have.
- Individualized alignment is challenging due to scalability issues, requiring repetitive acquisition of preference data and retraining models for each individual's preferences.
- LLMs are typically trained with uniform system messages, limiting their ability to generalize to diverse, unseen preferences.

We propose the following:

- **New paradigm for aligning to diverse preferences**: We examine how explicitly stating the user’s preference in the **system message** can guide the LLM’s behavior to align with the user’s intentions. We propose to train LLMs with diverse system messages, each representing an individual’s multifaceted preferences, to generalize to unseen system messages.
- **Multifaceted preference dataset**: We create <span class="sys-name">[Multifaceted-Collection](https://huggingface.co/datasets/kaist-ai/Multifaceted-Collection-SFT)</span>, a preference dataset with 196k unique system messages reflecting diverse values beyond generic helpfulness and harmlessness. The dataset spans 65k user instructions and includes various combinations of user preferences.
- **High-performance LLM**: We train a family of 7B LLMs on <span class="sys-name">Multifaceted-Collection</span> using instruction tuning and preference optimization methods, named <span class="sys-name">[Janus](https://huggingface.co/datasets/kaist-ai/janus-7b)</span>. When tested on prompts from 5 benchmarks ([AlpacaEval 2.0](https://huggingface.co/datasets/tatsu-lab/alpaca_eval), [FLASK](https://github.com/kaistAI/FLASK/blob/main/evaluation_set/flask_evaluation.jsonl), [Koala](https://github.com/arnav-gudibande/koala-test-set), [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench), and [Self-Instruct](https://github.com/yizhongw/self-instruct/blob/main/human_eval/user_oriented_instructions.jsonl)) enhanced with diverse system messages reflecting user preferences, <span class="sys-name">Janus</span> outperforms other models. <span class="sys-name">Janus</span> even outperforms LLaMA 3 8B Instruct on general response helpfulness benchmarks ([AlpacaEval 2.0](https://huggingface.co/datasets/tatsu-lab/alpaca_eval), [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench), [Arena Hard Auto v0.1](https://lmsys.org/blog/2024-04-19-arena-hard/)).

------

## Training on various system messages for alignment
{: .sys-img}
![janus_overview](/assets/img/janus_overview.png)  

Previous LLMs are trained with homogeneous system messages reflecting general helpfulness and harmlessness. We propose training LLMs with diverse system messages, each representing an individual’s multifaceted preferences, to generalize to unseen system messages. The model we train in this direction, <span class="sys-name">Janus</span> 7B, is adept at generating personalized responses for personalized system messages.

### Role of system messages

A preference is implicit in typical preference datasets, which provide only a pair of chosen and rejected response to each instruction. However, people have diverse preferences that are nuanced in different contexts and it is difficult to know what made one preferred compared to the other. To reduce the ambiguity, we conceptualize a *preference* as a detailed textual description of a quality that a desirable response should possess from an individual's lens. Based on this definition, we identify two key requirements for a model to reflect the diversity of human preferences and devise a strategy for each of it.

- **Multifacetedness**: Individual preferences are multifaceted; for instance, one might expect multiple aspects like applicability, complexity, variability, and ethics in one responses. To represent this diversity, we use a **hierarchical preference augmentation strategy**, starting from general dimensions and branching into specific subdimensions and preferences. We further combine preferences from different dimensions to effectively represent the complex interplay of values.

- **Explicitness**: To help models learn the nuances between preferred and rejected responses interpretably, we make preferences explicit in the **input via detailed system messages** preceding the instructions.

### Multifaceted preference dataset: 
{: .sys-img}
![data_construction](/assets/img/data_construction.png)  

The <span class="sys-name">Multifaceted-Collection</span> is a dataset for aligning LLMs to diverse human preferences, built by a novel construction approach to make preferences multifaceted and explicit. We acquire 65k instructions from five existing datasets ([Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar), [OpenHermesPreferences](https://huggingface.co/datasets/argilla/OpenHermesPreferences), [UltraFeedback-binarized-clean](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned), [Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), [Domain-Specific Preference dataset (DSP)](https://github.com/Linear95/DSP/blob/main/data)). For each instruction, preference descriptions are augmented from general to specific, allowing for multiple facets to branch out. Then we combine preferences from various dimensions into a system message to materialize these preferences as model input. Following the system message and instruction, a gold response is generated. We use GPT-4-Turbo for preference augmentation, system message generation, and gold response generation.

Here is an interative visualization of the dimensions, top-5 subdimensions in each dimension, and keywords of 5 randomly sampled preference descriptions in each subdimension. Click the elements to see how diverse preferences can be!

{% include_relative assets/img/preference_sunburst.html %}

### <span class="sys-name">Janus</span> models

Using [Mistral-7B-v0.2](https://huggingface.co/mistral-community/Mistral-7B-v0.2) as its base model, we train <span class="sys-name">Janus</span> models on <span class="sys-name">Multifaceted-Collection</span> using instruction tuning and preference optimization methods like DPO and ORPO. Visit [our HuggingFace collection](https://huggingface.co/collections/kaist-ai/system-message-generalization-6657b608280c926a3d0ec09c) for the full list of resources.

## Performance

### Multifacetedness

{: .img-left}
![human_comparison](assets/img/human_comparison.png)

{: .img-right}
![multifacetedness_benchmarks](assets/img/multifacetedness_benchmarks.png)

On benchmarks containing instructions paired with synthetic system messages and reference answers validated by human annotators, human evaluators confirm that <span class="sys-name">Janus</span> 7B outperforms Mistral 7B Instruct v0.2 and GPT models. When using LLMs as evaluators, <span class="sys-name">Janus</span> models consistently surpasses other models as well.

### Helpfulness

![helpfulness_benchmarks](/assets/img/helpfulness_benchmarks.png)

On benchmarks that evaluate general response helpfulness, <span class="sys-name">Janus</span> 7B excels across all helpfulness benchmarks relative to other models. This suggests that system message generalization not only supports the creation of personalizable LLMs but also acts as an effective novel method for improving alignment with what humans generally perceives as helpful.

### Harmlessness
![harmlessness_benchmark](assets/img/harmlessness_benchmark.png)

When evaluated on [RealToxicityPrompts](https://github.com/allenai/real-toxicity-prompts), <span class="sys-name">Janus</span> 7B shows significantly lower toxicity while achieving high fluency and diversity. These findings altogether underscore the effectiveness of training a LLM with diverse system messages in balancing diversity, helpfulness, and safety, making it robust and versatile.


------

## Demo
{% include_relative assets/demo/demo.html %}

## Bibtex
If you find our work useful in your work, please consider citing our paper:

<pre>
@misc{lee2024aligning,
      title={Aligning to Thousands of Preferences via System Message Generalization}, 
      author={Seongyun Lee and Sue Hyun Park and Seungone Kim and Minjoon Seo},
      year={2024},
      eprint={2405.17977},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>

------

{: .logos}
[![Logo of KAIST](/assets/img/kaist_logo.png)](https://kaist.ac.kr)
[![Logo of LKLab](/assets/img/lklab_logo.jpg)](https://lklab.kaist.ac.kr/)
[![Logo of CMU](/assets/img/cmu-wordmark-stacked-r.png)](https://www.cmu.edu/)

<!-- {: .center .acknowledgement}
This research was supported by the **KAIST-NAVER Hypercreative AI Center**. -->
