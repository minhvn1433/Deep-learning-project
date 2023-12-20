## Instructions

### RNN:
1. Setup
   - Open notebook in Kaggle.
   - Add `UIT-VSMEC` dataset to `kaggle/input`.
   - Add embeddings to `kaggle/input` as a dataset with this link: `https://public.vinai.io/word2vec_vi_words_300dims.zip`.
   - Add `RNN.pth` to `kaggle/input` for pretrained weights.
   - Set `pretrained_path`, `train_data`, `valid_data`, `test_data`, `word2vec_path` compatibly. 
   - Choose Tesla P100 for accelerator.
2. Options
   - Set the variables `TRAIN`, `PRETRAINED`, `TEST`, `INFER` as desired.
   - Choose "Run all"

### ...:
1. **Download dataset and weights**:
   - Download `train.xlsx`, `valid.xlsx` and `test.xlsx`
   - Download weights: https://drive.google.com/drive/folders/1GzTExOHED4goougRlmKtsR5QLBjtcvbT

2. **Open the notebook**:
   - Click `Open in Colab`

3. **Set up GPU acceleration**:
   - Access menu bar
   - Select `Runtime`
   - Choose `Change runtime type`
   - Under `Hardware accelerator`, select `T4 GPU`
   - Click `Save`

4. **Upload dataset and weights**:
   - Access left sidebar
   - Select `Files`
   - Upload `train.xlsx`, `valid.xlsx` and `test.xlsx`
   - Upload weights
