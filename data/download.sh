#!/bin/bash

git lfs install
git clone https://huggingface.co/datasets/H34lthy/Isotropy
mv Isotropy/*.jsonl ./
rm -rf Isotropy
