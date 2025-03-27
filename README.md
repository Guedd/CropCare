# CropCare AI-Powered Crop Disease Monitoring 

This project will develop an innovative, user-friendly mobile phone application for early detection of potato diseases, a major concern for Finnish potato farmers that could be exacerbated by climate change.

The project will leverage the [National Plant Phenotyping Infrastructure (NaPPI)](https://www.helsinki.fi/en/infrastructures/national-plant-phenotyping)
imaging facilities within the [University of Helsinki (UH)](https://www.helsinki.fi/en/), the advanced machine learning capabilities from the [University of Oulu (UO](https://www.oulu.fi/en)), and the experience of Pyhäjärvi-Institute as an advisory partner.

# Project and Tools
## CropCare - Large Language Model
Large Language Model is a AI model capable of understanding the human language, analyze context, generate text, answer question, and more. CropCare implement the LLM technics to provide AI chatbot Assistante service, where we used a language model called [PLLaMa](https://arxiv.org/abs/2401.01600) as our language model, this model have been trained and finetuned on 1.5 million plant science scientific papers. but the model have only trainned on instructions and info extracted from papers. we have laverage the use of the model to be used on chat mode, and by training the PLLaMa on different Question & Answer datasets, without lossing the knowladge of the model. the next step with our version of PLLaMa is to be trained on different potato diseases and treatements files.

## CropCare - Vision Language Model
A VLM or a Vision Language Model is a model that can deals with images and text as input, for example a model that can descripe an image. For CropCare we are integratting a [LLaVa-Large Language and Vision Assistant](https://llava-vl.github.io/). this large multimodal model that combines a vision encoder and Vicuna as Text decoder for general-purpose visual and language understanding. Vicuna is a model with the same configuration as PLLaMa both are LLama-2 models with a small modification in the model configuration. as for the Vision encoder the model uses (CLIP-Vit)[https://openai.com/index/clip/] model from (open AI Company)[openai.com](https://openai.com/)]. The CLIP model trained on 400 millions images from different datasets, which make it one of the best models created to use for image understanding and visual features extraction.

## Our Model
for our implementation we are aiming to re-config the LLaVa-Vicuna model to use our finetuned PLLaMa model as text generator insted of Vicuna, for this resoan we have generated our dataset, a combanitaton of image and text descriptions, to help train LLaVa-PLLaMa.

## Dataset
