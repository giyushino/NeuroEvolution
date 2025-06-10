# NeuroEvolution
Testing how effective "natural selection" is in producing accurate neural networks 


## To Do 
- [x] Write CNN, ViT (pytorch + jax for both)
    - Putting jax off for now, not exactly sure how to speed up model initialization, which is necessary for evolutionary stuff
- [x] Set up 2 datasets 
    - Google Doodle + real images -> i'll do real later
- [x] Test if parsing the raw jsonl is faster than datasets (i think it probably is)
    - Doesn't matter I think, both are fast enough 
- [ ] Rank the classes by how similar they are and see if we can get the ordering correct, ie dragon and crocodile should be similar to each other 
    - Maybe save embeddings 
- [ ] Set up normal training pipeline 
- [ ] Train all 4 models 
- [ ] Create function to compare weights (cosine similarity) 
- [ ] Write evolutionary algorithm
- [ ] Test model robustness (add noise)
- [ ] Create dataset with predators, see if model learns to differentiate between different animals 
    - Like if we know ducks don't hurt us but crocodiles + lions will, is there any point to learning the difference between croc/lion
- [ ] See what features are the most important 
    - Grad Cam, see if this allows us to determine what layers mutations should affect the most  
    - hooks, check out some other mechanistic interpretability techniques 
- [ ] Plot loss landscape
- [ ] Front end maybe? 
- [ ] Animations? 

