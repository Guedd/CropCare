
# Data Directory

This directory contains the generated intermediate/output files for the images.

The expected structure within the `data` directory is:</br>
data/</br>
└── prompts/# Contains generated data files(CSV index, final JSON dataset). </br>

---

## `prompts/`

* **Purpose:** Stores intermediate and final output data files generated during the workflow.
* **Key Files:**
    * `image_indexing.csv`: Initially created by a script with image paths/IDs. It is subsequently updated to include generated text descriptions (prompts and responses) obtained via OpenAI API calls. This file serves as a central index used by multiple scripts.
    * Final JSON Dataset File: `Pllava_detailed_potato_data.json` Generated from the final `image_indexing.csv`. Contains the image paths and conversation data formatted for specific downstream tasks (like LLaVA model training).

