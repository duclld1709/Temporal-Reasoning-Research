Dataset from the paper "Improving Event Duration Question Answering by Leveraging Existing Temporal Information Extraction Data"  
Original Repository: felixgiov/UDST-DurationQA

### Overview of the dataset
We recast our data from the [UDS-T dataset](http://decomp.io/projects/time/).
We provide the train/dev/test split as specified in the paper under `data/`.
In each file, there are lines of tab-separated data, each line representing an instance of a question-answer pair.
Each line contains the following information:

`sentence \t  question \t  answer \t label`

 * **sentence**: a context sentence where the question is based on.
 * **question**: a question asking about the duration of an event in the sentence.
 * **answer**: a potential answer to the question.
 * **label**: whether the answer is a plausible answer. "yes" indicates the answer is plausible, "no" otherwise.

### Citation
If you use this dataset, please cite our paper ["Felix Giovanni Virgo, Fei Cheng, Sadao Kurohashi. Improving Event Duration Question Answering by Leveraging Existing Temporal Information Extraction Data. Proceedings of the 13th International Conference on Language Resources and Evaluation (LREC), (2022)"](https://aclanthology.org/2022.lrec-1.473/)
