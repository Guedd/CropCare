# CropCare - Dataset
## PLLaMa 
CropCare is a model built on top of [PLLaMa](https://arxiv.org/abs/2401.01600), the Large Language Model fine-tuned on 1.5 million plant science papers. After testing the model, it showed strong knowledge in plant sciences and related fields. We further fine-tuned the model using two different Question & Answer datasets from the [HiggingFace](https://huggingface.co/) website:
* [Meithnav/Agriculture](https://huggingface.co/datasets/meithnav/agriculture) 
* [Mahesh2841/Agriculture](https://huggingface.co/datasets/Mahesh2841/Agriculture)

Both datasets were created by [ChatGPT](https://openai.com/index/chatgpt/) and contain Q&A data in various fields of agriculture and farming. The datasets include instructions for the LLM to follow, along with questions and answers.

## PLLaMa - Next Finetuning
The next step is to finetune the PLLaMa model on a collected dataset derived from various papers and books in the field of potato diseases. For data preprocessing, we apply optical character recognition (OCR) techniques and services to extract text from PDF files.

## LLaVa-PLLaMa 
For this model, we used ChatGPT to create an Image-Text dataset. [A novel dataset of potato leaf disease in uncontrolled environment](https://www.sciencedirect.com/science/article/pii/S2352340923009861) is a new dataset for potato diseases, containing images classified by the names of disease families such as `Healthy`, `Virus`, `Fungal`, and more.

Following the LLaVa team’s approach, they used ChatGPT to generate text descriptions for the images they used to train LLaVa. You can read more about how and what prompts we used to generate the descriptions of the images in the "[Images](https://github.com/Guedd/CropCare/blob/main/Data/Images/README.md)" folder.
