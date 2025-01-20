# llm-from-scratch

This repository is the result of my studies of Large Language Models (LLMs).

At the moment, there are in this repository two notebooks (`shakespeare.ipynb` and `tolkien.ipynb`) where I have trained a LM, which I kindly baptized as `Small Language Model' (SLM), on respectively the full texts of Shakespeare (as in Andrej Karpathy's lectures) but also on the full texts of the Lord Of The Rings trilogy (a step further from Karpathy's lectures).

Following the tutorials down below, `SLM` uses only the characters of the text (i.e. a couple dozens of distinct characters) as its vocabulary tokens (instead of more than 10K tokens GPT actually uses). However, it's fun and surprising to see the results of the output after training.

If you want to reproduce my studies, I have collected the following (non-exhaustive list but free online) material:

- I. Goodfellow, Y. Bengio, and A. Courville, ``Deep Learning'' (2016). http://www.deeplearningbook.org

- G. James, D. Witten, T. Hastie, R. Tibshirani, and J. Taylor, ``An Introduction to Statistical Learning'' (2023). https://www.statlearning.com/

- Stanford's ``Natural Language Processing with Deep Learning'' course (full lectures):
https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4

- 3b1b's ``Neural Network'' series (higher-level explanations):
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

- StatQuest's ``Neural Network / Deep Learning'' series (lower-level explanations):
https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=2

- Andrej Karpathy's ``From zero to hero'' tutorials (hands-on lectures):
https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ