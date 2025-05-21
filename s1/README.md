# Replicating s1-style results in inspect

## Split 1: baseline Qwen2.5-32B-Instruct on MATH in inspect
- Target time: 1:00
- Start time: 7:36
- Notes:
    - inspect makes this very easy
    - runpod needs a storage volume to download qwen 32B to disk
    - volume is /dev/shm, need to point hf hub there with "export HF_HOME=/your/new/path"
    - getting some GPU memory full errors despite 95G VRAM and only a 32B model, which should be 64G?
    - gonna try vllm inference to see if it's a problem with inspect or something
    - I also need to figure out how to get more than one GPU and a memory volume together
    - started this split too close to bedtime, calling it for mow.

- Break time: 7:23
- Resume time:
- End time:
- Split time:
- Debrief notes: