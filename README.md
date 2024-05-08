# PropositionExtraction

This repository hosts the implementation and accompanying resources for the paper "Propositional Extraction from Natural Speech in Small Group Collaborative Tasks" by Videep Venkatsha, Abhijnan Nath, Ibrahim Khebour, Avyakta Chelle, Mariah Bradford, Jingxuan Tu, James Pustejovsky, Nathaniel Blanchard, and Nikhil Krishnaswamy (appearing at EDM 2024). 

## Table of Contents
- [Installation](#installation)
- [Training](#training)
- [Models](#models)

## Installation

To set up your environment to run the code, you will need Python and the packages listed in `requirements.txt`.

### Prerequisites

- Python 3.8 or above
- pip

### Installing Dependencies

Clone the repository and install the required Python packages using:

```bash
git clone https://github.com/csu-signal/PropositionExtraction.git
cd PropositionExtraction
pip install -r requirements.txt
```
## Training
```bash
python XE_training.py <nameOfResultsFolder> <model>
```
Replace <nameOfResultsFolder> with the name of the folder where you want to save your results and <model> with the model you want to train.


## Available Models

You can train the following models using this script:

- `bert-base-uncased`
- `roberta-base`
- `allenai/longformer-base-4096`
