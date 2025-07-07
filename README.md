# NeuroEvolution
Testing how effective "natural selection" is in producing accurate neural networks 

## Set Up
After this, install [the proper version of Pytorch with GPU support for your device.](https://pytorch.org/get-started/locally/)
I'm using CUDA 12.8
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
We also need to install the GPU version of Jax 
```sh
pip install --upgrade "jax[cuda12]" 
```
Then to properly use this repo 
```sh
pip install -e .
```



## To Do 
- [x] Write CNN, ViT (pytorch + jax for both)
    - Putting jax off for now, not exactly sure how to speed up model initialization, which is necessary for evolutionary stuff
- [x] Set up 2 datasets 
    - Google Doodle + real images -> i'll do real later
- [x] Test if parsing the raw jsonl is faster than datasets (i think it probably is)
    - Doesn't matter I think, both are fast enough 
- [ ] Rank the classes by how similar they are and see if we can get the ordering correct, ie dragon and crocodile should be similar to each other 
    - Maybe save embeddings 
- [x] Set up normal training pipeline 
    - know it works for cnn and linear, idk about vit yet
- [ ] Train all 4 models 
- [x] Create function to compare weights (cosine similarity) 
- [x] Write evolutionary algorithm
    - for some reason converging super slowly, write it again
- [ ] Test model robustness (add noise)
- [ ] Create dataset with predators, see if model learns to differentiate between different animals 
    - Like if we know ducks don't hurt us but crocodiles + lions will, is there any point to learning the difference between croc/lion
    - study the embeddings ig
- [ ] See what features are the most important 
    - Grad Cam, see if this allows us to determine what layers mutations should affect the most  
    - hooks, check out some other mechanistic interpretability techniques 
- [ ] Plot loss landscape
- [ ] Front end maybe? 
- [ ] Animations? 

