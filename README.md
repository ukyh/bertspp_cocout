## Improved BERTScore for image captioning evaluation
<!-- Implementation of paper: Improving Image Captioning Evaluation by Considering Inter References Variance (ACL2020) -->

The code for `improved BERTScore` evaluation of our paper, **[Switching to Discriminative Image Captioning by Relieving a Bottleneck of Reinforcement Learning](https://github.com/ukyh/switch_disc_caption.git)** (WACV 2023).

### Acknowledgment
The code is based on [improved-bertscore-for-image-captioning-evaluation](https://github.com/ck0123/improved-bertscore-for-image-captioning-evaluation).
We thank the authors of the repository.


## Setup

```bash
git clone https://github.com/ukyh/bertspp_cocout.git
cd bertspp_cocout

conda create --name bertspp python=3.6
conda activate bertspp

pip install -r requirements.txt

# Test run
python -u run_metric_custom.py --file samples --dir example	
```


## Downloads

```bash
mkdir data; cd data
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
rm -f caption_datasets.zip

mkdir coco_caps
mkdir eval_results
```

Then, download [cocotalk_disc_text.zip](https://drive.google.com/file/d/12LY3FzL_zYHDUzp_pez9fvtA3LaATCZL/view?usp=sharing) and unzip it into `data/`:  
`unzip cocotalk_disc_text.zip -d data/`


## Run

Copy the output files to evaluate from [switch_disc_caption](https://github.com/ukyh/switch_disc_caption) (the files under `eval_results`).  
Then, run the following commands.

```bash
cd bertspp_cocout
conda activate bertspp
export PYTHONPATH=$PYTHONPATH:`pwd`

# NOTE: the end of the file name has to be "_val.json" or "_test.json"
ID=sample_test
python -u proc_bert_score_pp.py --hyp eval_results/${ID}.json
python -u run_metric_custom.py --file ${ID} --dir coco_caps
```


## Reference

If you find the paper or this code useful, please consider citing:

```
@inproceedings{honda2023switch,
  title={Switching to Discriminative Image Captioning by Relieving a Bottleneck of Reinforcement Learning},
  author={Honda, Ukyo and Taro, Watanabe and Yuji, Matsumoto},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
}

@inproceedings{yi2020improving,
  title={Improving image captioning evaluation by considering inter references variance},
  author={Yi, Yanzhi and Deng, Hangyu and Hu, Jinglu},
  booktitle={ACL},
  year={2020}
}
```

<!-- ## Usage:
Recently, this repo provides two metrics ('with BERT' and 'simple')

* python3 run_metric.py    
    
* python3 run_metric_simple.py 

## example data:

example/example.json (you can modify this file for your own datasets)   



Fields explanation:  
* "refs": reference captions (each sample 5 references)    
* "cand": candidate caption (each sample 1 candidate)
* "refs_hid": contextual embeddings of reference captions
* "cand_hid": contextual embeddings of cand captions
* "mismatch": mismatches marks computed from all of reference captions
* "metric_result": scores on our metric  

  
NOTE:   
we also provide Flickr 8K Expert Annotation file with our format 'example/flickr.json'  
you can easily reproduce our result following run_metric.py lines 223-235. -->


<!-- ## Dependencies:
pytorch-pretrained-bert==0.6.2 (old version of [huggingface/transformers](https://github.com/huggingface/transformers))     
torch==0.4.1  
bert_score==0.1.2 (already in this repo) -->


