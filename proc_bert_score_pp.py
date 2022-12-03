import json
import os
import argparse


DATASET = "data/dataset_coco.json"
COCOTALK = "data/cocotalk.json"

parser = argparse.ArgumentParser()
parser.add_argument('--hyp', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='coco_caps')
args = parser.parse_args()

hyp_fname = os.path.splitext(os.path.split(args.hyp)[-1])[0]
eval_split = hyp_fname.split("_")[-1]
assert eval_split in {"test", "val"}

dataset = json.load(open(DATASET))
cocotalk = json.load(open(COCOTALK))
_hyp = json.load(open(args.hyp))
hyp = {}
if "imgToEval" in _hyp:
    for imgid in _hyp["imgToEval"]:   # str(image_id)
        hyp[int(imgid)] = " ".join(_hyp["imgToEval"][imgid]["caption"].split())
else: # Oscar
    for item in _hyp:
        hyp[int(item["image_id"])] = " ".join(item["caption"].split())

# Split
val = {}
test = {}
for split_dict in cocotalk["images"]:
    imgid = split_dict["id"]
    if split_dict["split"] == "test":
        test[imgid] = []
    elif split_dict["split"] == "val":
        val[imgid] = []
    else:   # train or restval
        pass
assert len(test) == len(val) == len(hyp)

# # Vocab
# word_to_ix = {}
# for ix, word in cocotalk["ix_to_word"].items():
#     word_to_ix[word] = int(ix)

# Convert
train_freq = []
for img_dict in dataset["images"]:
    imgid = img_dict["cocoid"]
    cap_list = []
    for cap_dict in img_dict["sentences"]:
        cap = " ".join(cap_dict["tokens"])
        cap_list.append(cap)
    if eval_split == "test" and imgid in test:
        test[imgid] = cap_list
    elif eval_split == "val" and imgid in val:
        val[imgid] = cap_list
    else:   # train or restval
        pass

sorted_test = sorted(test.items(), key=lambda x:x[0]) # [(imgid, [caps]), ...]
sorted_val = sorted(val.items(), key=lambda x:x[0])
sorted_hyp = sorted(hyp.items(), key=lambda x:x[0])
for i, (imgid, _) in enumerate(sorted_hyp):
    if eval_split == "test":
        assert imgid == sorted_test[i][0]
    else:
        assert imgid == sorted_val[i][0]

MAX_REF_TEST = max([len(v) for k, v in sorted_test])
MAX_REF_VAL = max([len(v) for k, v in sorted_val])
if eval_split == "test":
    MAX_REF = MAX_REF_TEST
else:
    MAX_REF = MAX_REF_VAL

if eval_split == "test":
    sorted_ref = sorted_test
else:
    sorted_ref = sorted_val

out_dict = {}
for i, ((rid, caps), (hid, cap)) in enumerate(zip(sorted_ref, sorted_hyp)):
    out_dict[i] = {"refs":caps, "cand":[cap]}

out_fname = "{}/{}.json".format(args.out_dir, hyp_fname)
with open(out_fname, "w") as outfile:
    json.dump(out_dict, outfile, indent=4)
print("Created coco file: {}".format(out_fname))
