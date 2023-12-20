# Natural-Language-Processing
This repository contains notebooks related to Natural Language Processing HuggingFace tutorial.

## History of Transformers
The Transformer architecture emerged in June 2017 with a primary focus on translation tasks. Subsequently, several influential models were introduced, including:

- GPT in June 2018: The initial pretrained Transformer model, excelling in fine-tuning across various NLP tasks, achieving state-of-the-art results.
- BERT in October 2018: A substantial pretrained model aimed at generating improved sentence summaries.
- GPT-2 in February 2019: An enhanced and larger version of GPT, initially withheld from public release due to ethical concerns.
- DistilBERT in October 2019: A distilled variant of BERT, boasting 60% faster speed, 40% lighter memory usage, while maintaining 97% of BERT's performance.
- BART and T5 in October 2019: Large pretrained models using the original Transformer architecture.
- GPT-3 in May 2020: A larger iteration of GPT-2 demonstrating proficiency across diverse tasks without fine-tuning, employing zero-shot learning.

Broadly, these models fall into three categories: 
- GPT-like (auto-regressive Transformer models)
- BERT-like (auto-encoding Transformer models)
- BART/T5-like (sequence-to-sequence Transformer models)

## Self-Supervised Learning and Transfer Learning in Transformers

All major Transformer models like GPT, BERT, BART, and T5 have undergone training as language models. This training involves using large volumes of raw text in a self-supervised manner, where the model learns from its inputs without human-provided labels.

While these language models gain a statistical understanding of the language, they aren't initially practical for specific tasks. Hence, they undergo transfer learning, a process involving fine-tuning the model using human-annotated labels for a particular task.

## Challenges with Model Size and Pretraining

Transformers are generally large models. Although outliers like DistilBERT exist, enhancing performance usually involves increasing model sizes and the volume of pretrained data. However, training large models demands significant data and computational resources, resulting in time-consuming processes and environmental impact.

## The Importance of Sharing Pretrained Models

Given the costs associated with training large models from scratch, sharing pretrained model weights becomes crucial. This approach reduces overall computational costs and environmental impact within the community, facilitating knowledge building upon existing models.

## Pretraining vs. Fine-Tuning

Pretraining involves training a model from scratch with randomly initialized weights on vast datasets, often taking several weeks. In contrast, fine-tuning occurs after a model has been pretrained. By leveraging a pretrained model and training it further on a task-specific dataset, fine-tuning benefits from the pretrained model's knowledge and requires less data, time, and resources.

## Advantages of Fine-Tuning Over Training from Scratch

Fine-tuning offers advantages over training directly for a task due to the pretrained model's existing knowledge, needing less data and resources, and being quicker to achieve good results. It allows easier iteration over various fine-tuning schemes, leading to better outcomes than training from scratch-except when substantial data is available.

In essence, leveraging a pretrained model and fine-tuning it closely aligns with our task, optimizing results while minimizing time, data, financial, and environmental costs.

## Components of the Transformer Model

The Transformer model comprises two key blocks:

- **Encoder:** Focuses on building representations (features) from input, optimizing the model for understanding the input.
- **Decoder:** Utilizes the encoder's representation along with other inputs to generate a target sequence, emphasizing the model's capability to generate outputs.

These parts can be utilized independently based on the task at hand:

- **Encoder-only models:** Suitable for tasks requiring input understanding, like sentence classification or named entity recognition.
- **Decoder-only models:** Ideal for generative tasks, such as text generation.
- **Encoder-decoder or sequence-to-sequence models:** Effective for generative tasks needing both input and output, such as translation or summarization.

## Attention Layers in Transformers

A distinctive feature of Transformer models is their inclusion of attention layers. The paper introducing the Transformer architecture even bore the title "Attention Is All You Need." While delving into attention layer intricacies comes later in this course, presently, grasp that these layers guide the model to focus on specific words in a sentence, giving them priority (or ignoring them) in word representations.

For instance, in the task of translating from English to French, the model needs to consider adjacent words like "You" to accurately translate "like," given the verb conjugation in French depends on the subject. Similarly, when translating "this," the model should consider the word "course" due to gender-specific translations in French. Extraneous words in the sentence don’t contribute to translating these specific words. This concept extends to any natural language task: individual words derive meaning from context, which can involve preceding or following words.

## Encoder Models in Transformer Architecture

Encoder models exclusively utilize the encoder of a Transformer model. 
- In each phase, the attention layers have access to all words in the original sentence, displaying "bi-directional" attention.
- These models are commonly referred to as auto-encoding models.
- The pretraining process for these models typically involves manipulating a given sentence (like randomly masking words) and challenging the model to reconstruct or retrieve the original sentence.
- Encoder models excel in tasks requiring a comprehensive understanding of the entire sentence.
- They are well-suited for tasks like sentence classification, named entity recognition, word classification, and extractive question answering.
- Notable models belonging to this category include:
  - ALBERT
  - BERT
  - DistilBERT
  - ELECTRA
  - RoBERTa

## Decoder Models in Transformer Architecture

Decoder models exclusively utilize the decoder of a Transformer model. 
- In each stage, the attention layers focus solely on the words preceding a given word in the sentence, leading to a "auto-regressive" behavior.
- Pretraining for decoder models typically involves predicting the subsequent word in a sentence.
- Decoder models excel in tasks primarily focused on text generation.
- Prominent models falling under this category include:
  - CTRL
  - GPT
  - GPT-2
  - Transformer XL

## Encoder-Decoder Models in Transformer Architecture

Encoder-decoder models, also known as sequence-to-sequence models, leverage both the encoder and decoder components in the Transformer architecture. 
- During each stage, the encoder's attention layers can access all words in the original sentence, while the decoder's attention layers focus solely on words preceding a given word in the input.
- Pretraining for these models often involves more complex objectives compared to encoder or decoder models. For example, T5 is pretrained by replacing random spans of text with a single mask special word and predicting the text replaced by this mask word.
- Sequence-to-sequence models excel in tasks centered on generating new sentences based on a given input.
- Such tasks include summarization, translation, or generative question answering.
- Noteworthy models belonging to this category include:
  - BART
  - mBART
  - Marian
  - T5

## Bias and Limitations

While pretrained or fine-tuned models are potent tools, they do have limitations. 
- One significant aspect is that for extensive pretraining, researchers often scrape diverse content from the internet, encompassing both high-quality and problematic data.
- It is crucial to bear in mind that when utilizing these tools, the original model might generate content embodying biases like sexism, racism, or homophobia.
- Fine-tuning the model on our data won't inherently eliminate these biases.

# Chapter 1: Transformer Models
- Explored the `pipeline()` function in 🤗 Transformers for various NLP tasks.
- Demonstrated the process of locating and utilizing models from the Hub.
- Utilized the Inference API for direct model evaluation in the browser.
- Discussed the core functionalities of Transformer models, highlighting transfer learning and fine-tuning.
- Emphasized the flexibility in employing the full architecture, encoder, or decoder based on the task at hand.
- **Encoder Models:** ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa for tasks like sentence classification, named entity recognition, and extractive question answering.
- **Decoder Models:** CTRL, GPT, GPT-2, Transformer XL for text generation.
- **Encoder-Decoder Models:** BART, T5, Marian, mBART for tasks such as summarization, translation, and generative question answering.
- 
# Chapter 2: Using 🤗 Transformers
- Acquired foundational knowledge of a Transformer model's fundamental components.
- Explored the intricacies of a tokenization pipeline and its vital role.
- Practically implemented a Transformer model, grasping its practical usage.
- Utilized a tokenizer efficiently to convert text into model-understandable tensors.
- Configured a seamless integration of tokenizer and model for predictions.
- Recognized the limitations of input IDs and gained insights into attention masks.
- Explored and experimented with versatile and customizable tokenizer methods.


# Acknowledgements
- Hugging Face: [The Hugging Face Course 2022](https://huggingface.co/learn/nlp-course).

