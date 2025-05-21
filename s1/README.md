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

    - can I use together.ai
        - serve qwen 32b
        - compatible with inspect
        - support finetuning?
    - debug runpod?
    - switch to lambda

    - switched to together
    - gonna start prep for next split while eval runs

- Break time: 7:23
- Resume time: 7:16
- End time: 7:49
- Split time: 1:20
- Debrief notes:
    - inspect part of this was very easy, just copy paste
    - if I had used togetherai from the start this would've been a fifteen minute split
    - there was no real reason for me to use runpod
    - but once I was using runpod...
        - not getting a storage volume the first time slowed me down
        - not using hf_xet probably slowed me down
        - gpu mem filling up debugging slowed me down
            - not sure if it's worth going back and figuring out how to solve this?
        - didn't get to it but memory+multi-gpu provisioning thing would have slowed me down
            - ditto
    - I think main lesson is don't touch GPUs unless you're doing internals stuff
    - oh and uh Qwen-2.5-Coder-32B-Instruct got 54.4% on MATH_500
        - I'll circle back and fix AIME later

## Split 2: implement budget forcing, check MATH
- Target time: 1:00
- Start time: 1:30
- Notes:
    - took a break for 24m to chat
    - how do I implement budget forcing?
        - I could do it in a solver, but solvers have no way to continue generation from within a message. I'd have to do something like
        take <message1>2+2=5.</message1>, edit it to <message1>2+2=5. wait...</message1> and then generate <message2>. Which is dist shift
        - Do I have to make my own version of model.generate()? That's a bit deep in the guts of inspect, a bit hard to get it integrated properly
        - Maybe I can make my own provider? If that lets me just call togetherai myself and plug it into Inspect, that'd be ideal
    - Inspect doesn't seem like it can run Together.AI finetunes either. Maybe I drop inspect for now
    - Trying that
        - retook baseline, 42.2%
            - checking QwQ to see if it matches paper
            - ok nvm rate limit way too slow. Just gonna move on
    - Forcing completions to 2048
        - Accuracy:
- End time:
- Split time:
- Debrief notes: