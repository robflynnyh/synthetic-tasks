- install via `pip install -e .` after cloning and navigating to the repo's main directory
- Navigate to any of the tasks folders to run a task via train.py

# Tasks:
## words-in-location-counting-task

Simple synthetic task to test bidirectional sequence models!

decription:
Given a sequence of sentences S \ni {s_0, ..., s_n} count the total number of words in sentence s_n, s_{n-2}, s_{n+2}. Sentences are made up of a random sequence of words of varying length.

- For longer sentence lengths the model requires a curriculum learning approach
- Tested for sentences up to 30 words with inputs that contain 100 sentences, the model can solve this although it takes a bit of time
- Here's the weights and biases run for this experiment! https://wandb.ai/wobrob101/word-count/runs/j15svvuj/workspace?nw=nwuserwobrob101
