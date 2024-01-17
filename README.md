# MARIO

This is the official repository for the paper [MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline](http://arxiv.org/abs/2401.08190). We release our codes.


MARIO REACT Corpus coming soon.


|     	| Base Model: Llemma                                           	| Outcome Value Model                                                    	| 
|-----	|---------------------------------------------------------------	|---------------------------------------------------------------------------	|
| 7B  	|[🤗](https://huggingface.co/MARIO-Math-Reasoning/MARIO-7B-v1)[🤖](https://huggingface.co/MARIO-Math-Reasoning/MARIO-7B-v1)MARIO-7B| [🤗](https://huggingface.co/MARIO/MARIO-OVM-7B)[🤖](https://www.modelscope.cn/models/damo/MARIO-OVM-7B)MARIO-OVM-7B|
| 34B 	|[🤗](https://huggingface.co/MARIO-Math-Reasoning/MARIO-34B-v0)[🤖](https://www.modelscope.cn/models/damo/MARIO-34B)MARIO-34B||


## Performance
We demonstrate the results of our MARIO-7B and MARIO-34B as follows:

| **Model**             	| **Decoding** 	| **GSM**  	| **MATH** 	| **OCWCourse** | **Gaokao-2023-ME** | 
|---------------------------|---------------|-----------|-----------|-----------|-----------|
| MARIO-OVM-7B + OVM@20	| **Hybrid**   	| **83.6** | **60.6**    | 25.4 |	42.9 |
| MARIO-7B + OVM@20  	| **Hybrid**   	| 82.9  	| 59.1  | **28.3**   	| **45.2** 	|
| MARIO-OVM-7B       	| **Hybrid**   	| 74.5  	| 47.7 	    | 19.1   	|32.5   	|
| MARIO-7B             	| **Hybrid**   	| 70.1  	| 46.3 	    | 19.9  	|35.6   	|
| ToRA-Code-7B  	    | **Hybrid**   	| 72.6  	| 44.6  	| 4.8  	| 23.9	|
| MAmmoTH-Coder-7B  	    | **Hybrid**   	| 59.4  	| 33.4  	| 11.0  	| 15.3	|
| MathCoder-7B  	    | **Hybrid**   	| 67.8  	| 30.2 	| -  	|-   	|
| MetaMath-7B-Mistral       | **CoT**   	| 77.7  	| 28.2 	    | -      |-   	|
| OpenChat-3.5-7B           | **CoT**   	| 77.3 	    | 28.6 	    | -      |-   	|
| ChatGLM-3-6B              | **CoT**       | 72.3      | 25.7      | -  | - |

| **Model**             	| **Decoding** 	| **GSM**  	| **MATH** 	| **OCWCourse** | **Gaokao-2023-ME** | 
|---------------------------|---------------|-----------|-----------|-----------|-----------|
| MARIO-34B             	| **Hybrid**   	| 78.7  	| **53.1** 	    | **25.4**   	|**41.3**   	|
| ToRA-Code-34B  	    | **Hybrid**   	| 80.7  	| 50.8  	| 5.5  	|31.7	|
| MAmmoTH-Coder-34B  	    | **Hybrid**   	| 72.7  	| 43.6  	| 14.0  |25.2 	|
| MathCoder-34B  	    | **Hybrid**   	| 81.7  	| 45.2  	| -	| -	|
| DeepSeek-Coder-33B        | **PoT**   	| 60.7   	| 29.1 	    | -     |-	|
| QWen-72B                  | **CoT**       | 78.9      | 35.2      | -         |-   	|

## **Installation**

Clone this repository and install the required packages:

```bash
git clone https://github.com/MARIO-Math-Reasoning/MARIO.git
cd MARIO
pip install -r requirements.txt
pip install -e ./math_evaluation
```

## **Training and Inference**

Coming Soon

### **Quick Start**
To play with our model, run:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
```

### **Large-scale Evaluation with vllm**

To replicate the experimental results in our paper, run:

```bash

```

## Acknowledgements
- hiyouga's [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/)

## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@misc{liao2024mario,
      title={MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline}, 
      author={Minpeng Liao and Wei Luo and Chengxi Li and Jing Wu and Kai Fan},
      year={2024},
      eprint={2401.08190},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
