# Tasks:
## word-counting-task

Simple synthetic task to test bidirectional sequence models!

decription:
Given a sequence of sentences S \ni {s_0, ..., s_n} count the total number of words in sentence s_n, s_{n-2}, s_{n+2}. Sentences are made up of a random sequence of words of varying length.

- For longer sentence lengths the model requires a curriculum learning approach
- Currently testing if models can reliably solbe this task for sentences up to 30 words
