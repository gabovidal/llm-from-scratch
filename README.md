# llm-from-scratch

This repository is the result of my studies of Large Language Models (LLMs).

At the moment, there are in this repository two notebooks (`shakespeare.ipynb` and `tolkien.ipynb`) where I have trained a LM, which I kindly baptized as `Small Language Model' (SLM), on respectively the full texts of Shakespeare (as in Andrej Kaparthy's lectures) but also on the full texts of the Lord Of The Rings trilogy (a step further from Kaparthy's lectures).

Following Kaparthy's lectures, `SLM` uses only the characters of the text (i.e. a couple dozens of distinct characters) as its vocabulary tokens (instead of more than 10K tokens GPT actually uses). However, it's fun and surprising to see the results of the output after training.

If you want to reproduce my studies, I have collected the following (non-exhaustive list but free online) material:

- I. Goodfellow, Y. Bengio, and A. Courville, ``Deep Learning'' (2016). http://www.deeplearningbook.org

- G. James, D. Witten, T. Hastie, R. Tibshirani, and J. Taylor, ``An Introduction to Statistical Learning'' (2023). https://www.statlearning.com/

- Stanford's ``Natural Language Processing with Deep Learning'' course (full lectures):
https://youtu.be/rmVRLeJRkl4?si=AWpGsVXzAXwpSa1F

- 3b1b's ``Neural Network'' series (higher-level explanations):
https://youtu.be/aircAruvnKk?si=Q2RR4JWHG6KJoo0N

- StatQuest's ``Neural Network / Deep Learning'' series (lower-level explanations):
https://youtu.be/CqOfi41LfDw?si=cq8Ki9p_b-HqqEAN

- Andrej Kaparthy's ``From zero to hero'' series (hands-on lectures):
https://youtu.be/VMj-3S1tku0?si=crt46akN7XATCwhF
