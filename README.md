# CropCare AI-Powered Crop Disease Monitoring 

This project will develop an innovative, user-friendly mobile phone application for early detection of potato diseases, a major concern for Finnish potato farmers that could be exacerbated by climate change.

The project will leverage the [National Plant Phenotyping Infrastructure (NaPPI)](https://www.helsinki.fi/en/infrastructures/national-plant-phenotyping)
imaging facilities within the [University of Helsinki (UH)](https://www.helsinki.fi/en/), the advanced machine learning capabilities from the [University of Oulu (UO](https://www.oulu.fi/en)), and the experience of Pyhäjärvi-Institute as an advisory partner.

# Project and Tools
## CropCare - Large Language Model
A Large Language Model (LLM) is an AI model capable of understanding human language, analyzing context, generating text, answering questions, and more. CropCare implements LLM techniques to provide an AI chatbot assistant service. We use a language model called [PLLaMa](https://arxiv.org/abs/2401.01600, trained and fine-tuned on 1.5 million scientific papers in plant science. However, the model has only been trained on instructions and information extracted from these papers. We have optimized the use of this model for chat interactions by training PLLaMa on various Question & Answer datasets while retaining its domain-specific knowledge. The next step for our version of PLLaMa involves training it on comprehensive datasets covering different potato diseases and treatment methods.

## CropCare - Vision Language Model
A Vision Language Model (VLM) can process both images and text, such as generating textual descriptions of images. For CropCare, we are integrating  [LLaVa-Large Language and Vision Assistant](https://llava-vl.github.io/), a powerful multimodal model combining a vision encoder with Vicuna as a text decoder for general visual and language understanding. Vicuna has the same configuration as PLLaMa; both are based on LLama-2 models with slight modifications in their configurations. The vision encoder component of our VLM uses the [CLIP-Vit](https://openai.com/index/clip/) model developed by (openAI)[openai.com](https://openai.com/)]. The CLIP model was trained on 400 million images from diverse datasets, making it highly effective for image understanding and visual feature extraction.

## Our Model
for our implementation we are aiming to re-config the LLaVa-Vicuna model to use our finetuned PLLaMa model as text generator insted of Vicuna, for this resoan we have generated our dataset, a combanitaton of image and text descriptions, to help train LLaVa-PLLaMa.

## Dataset
