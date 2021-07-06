#!/bin/bash
mkdir -p "../models"
gsutil cp "gs://comp66090-storage/claimExtraction/finetuned-models/roberta-large (large-manual) 71.zip" "../models/roberta-large (large-manual) 71.zip"
unzip "../models/roberta-large (large-manual) 71.zip" -d "../models/roberta-large (large-manual) 71"
rm -f "../models/roberta-large (large-manual) 71.zip"