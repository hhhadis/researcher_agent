---
license: other
license_name: research-dataset-license
license_link: LICENSE
language:
- en
extra_gated_prompt: "You agree to abide by all terms of the Research-14K Dataset License, including proper attribution and restrictions on redistribution and commercial use. You also commit to using the dataset ethically and responsibly, refraining from any unlawful or harmful applications."
extra_gated_fields:
  First Name: text
  Last Name: text
  Country: country
  Affiliation: text
  Academic Status: 
    type: select
    options:
      - Student
      - Researcher
      - Professor
      - Industry Professional
      - Other
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research Ideation
      - Literature Review Assistance
      - Experimental Design Planning
      - Methodology Development
      - Draft Writing Practice
      - Research Validation
      - Hypothesis Generation
      - Reference Organization
      - Writing Improvement
      - Academic Training
      - Research Planning
      - Supplementary Tool
      - label: Other
        value: other
  Research Field:
    type: select
    options:
      - Machine Learning
      - Computer Vision
      - Natural Language Processing
      - Robotics
      - Other AI Fields
      - label: Other
        value: other
  geo: ip_location
  I agree to the terms and conditions of the Research-14K Dataset License: checkbox
  I agree to properly cite the Research-14K Dataset in any publications or projects: checkbox
  I will NOT use the dataset for any unlawful or unethical purpose: checkbox
  I understand that direct commercial use of the raw dataset requires explicit permission: checkbox
  I will NOT use the dataset to create or distribute harmful content: checkbox
  I will NOT use the dataset for creating academic papers without acknowledging the proper source: checkbox
extra_gated_button_content: Submit
size_categories:
- 10K<n<100K
---
# CycleResearcher: Automated Research via Reinforcement Learning with Iterative Feedback


HomePage: https://wengsyx.github.io/Researcher/

### Researcher-14K Dataset


The research-14k dataset is designed to capture both structured outlines and detailed main text from academic papers. The construction process involves three main steps:

#### 1. Data Collection and Preprocessing
We first compile accepted papers from major ML conferences (ICLR, NeurIPS, ICML, ACL, EMNLP, CVPR, and ICCV) from 2022 to 2024. Using Semantic Scholar (https://www.semanticscholar.org/), we:
- Retrieve ArXiv links and LaTeX-format files
- Collect a set of accept papers (from NeruIPS, ICLR, ICML, CVPR and ACL)
- Apply rule-based filtering to remove:
  - Comments (%) 
  - Acknowledgments
  - Other irrelevant content

#### 2. Background Enhancement
To ensure comprehensive research background, we:
- Use Semantic Scholar API to retrieve cited works
- Extract abstracts from citations
- Add citation context to bib files

#### 3. Structure Organization
For better research process understanding:
- Organize main body into structured outlines
- Separate papers into distinct sections
- Use Mistral-Large-2 model for outline extraction
- Follow outline structure as shown in Figure 1
- Concatenate outlines with corresponding sections

The final dataset comprises:
- Input: Detailed reference files
- Output: Paper outlines and main text

This process creates a complete fine-tuning dataset that captures both the content and structure of academic papers.

### Example

```
{
	"paperId": '59f6de04d1dc37...', # the Paper Id of Semantic Scholar
	"title": 'CofiPara: A Coarse-to-fine-paradigm for Multimodal ...',
	"abstract": 'Social media abounds with multimodal sarcasm, ...',
	"venue": 'Annual Meeting of the Association for Computational Linguistics',
	"year": 2024,
	"references": [{'paperId':'d98aa44f79fe...','title':'GOAT-Bench...','abstract':'The ...'},...],
	"arxiv":'2405.00390',
	"sections":[['introduction','Sarcasm, a prevalent from of figurative...'],['Related Work','...'],...],
	"figure":['intro_1.pdf',...],
	"messages":[{'role':'system','content':'...'},{'role':'user','content':'...'}]
}
```


## Using Researcher-14K
You can easily download and use the arxiver dataset with Hugging Face's [datasets](https://huggingface.co/datasets) library.
```py
from datasets import load_dataset

dataset = load_dataset("WestlakeNLP/Research-14K") 
print(dataset)
```

Alternatively, you can stream the dataset to save disk space or to partially download the dataset:
```py
from datasets import load_dataset

dataset = load_dataset("WestlakeNLP/Research-14K", streaming=True)
print(dataset)
print(next(iter(dataset['train'])))
```


## Model Specifications

|       Model Name        |                 Pre-training Language Model                  |                           HF Link                            |
| :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CycleResearcher-ML-12B  | [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-12B) |
| CycleResearcher-ML-72B  | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-72B) |
| CycleResearcher-ML-123B | [Mistral-Large-2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-123B) |

The CycleResearcher models are trained in Researcher-14K.

## Model Info

The CycleResearcher model series includes two main variants:

1. **ML Series**: Specifically trained for machine learning research, including computer vision (CV), natural language processing (NLP), and multimedia (MM)
2. **Science Series**: Extended to broader scientific domains (Coming soon)

All models have undergone extensive training on our Research-8k dataset and are optimized using the CycleReviewer feedback loop. According to our license, **all models and their derivatives cannot be used for generating papers without proper disclosure of AI assistance.** We also provide FastDetectGPT-based tools to detect potential misuse of these models.

**Model Release Date**: October 2024  
**Knowledge Cutoff Date**: October 2024

## CITE
```
@inproceedings{
weng2025cycleresearcher,
title={CycleResearcher: Improving Automated Research via Automated Review},
author={Yixuan Weng and Minjun Zhu and Guangsheng Bao and Hongbo Zhang and Jindong Wang and Yue Zhang and Linyi Yang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=bjcsVLoHYs}
}
```

### Open Source License

The code in this repository is open-sourced under the Apache-2.0 license. The model weights are open-sourced under the CycleResearcher-License.  The datasets are open-sourced under the Research-Dataset-License.