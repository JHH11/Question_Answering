# BERTforQA
Fine-tuning BERT for Question Answering System with SQuAD Dataset

## SQuAD Dataset
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers 
on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, 
or the question might be unanswerable.

## Preprocess
We wnat to build a QA system, which's **input** is a combination of question and paragraph, and **output** is a partition of paragraph.
However, there are some cases that their answers don't exist on corresponding paragraphs in SQuAD dataset. We select the cases that their answers 
exist on corresponding paragraphs and split into train dataset, validation dataset and test dataset at a ratio of 8:1:1. 

## Tips
1. **Answer always exist on paragraph.**

   Because of the limitation of `max_paragraph_len`, we need to reduce the length of tokenized paragraph. We should retain the answer in reduced paragraph
   cautiously.

2. **Split paragraphs to many windows in test phase**

   Unlike training phase, we can't know the position of answer in paragraph. We split paragraphs to many windows with size `doc_stride`.
   Let all windows execute **forward** in model and choose the best one which has the greatest sum of **start_logits** and **end_logits**.
   
3. **Automatic Mixed Precision**

   Some operations, like linear layers and convolutions, are much faster in `float16`. Other operations, like reductions, often require the dynamic range of `float32`. 
   Mixed precision tries to match each op to its appropriate datatype, which can reduce your network’s runtime and memory footprint.

4. **Learning Rate Schedules**


