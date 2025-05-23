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
        - Accuracy: 41.4%
    - Regardless of if it's correct, calling it here for this split
- End time: 4:30
- Split time: 2:36
- Debrief notes:
    - ok this was mad slow
    - I ended up abandoning inspect because it only exposes a messages interface
    - I spent a bunch of time waiting for MATH500 to run, but didn't really need to run all 500
    - Lots of time wasted doesn't mean lots of mistakes to point to
        - If at the outset of the whole project I'd aimed to just build the budget forcing loop and test its performance, using together, no inspect, and a test set of ~50, all this so far probably could've been done in like an hour tops

## Split 3: finetune on s1k and baseline+force on MATH
- Target time: 1:30
- Start time: 6:00
- Notes:
    - set up dataset
    - had to find a model available for both serverless inference and fine tuning
    - never mind finetuned models require dedicated endpoints anyway
    - edited rate limiter to deal with concurrent inference instead of RPM
    - big run is maybe ready to go? But it's time to go climbing so pin in this for now
- End time: 7:42    
- Split time: 1:42
- Debrief notes:
    - ok so now I know Together uses dedicated endpoints for finetune inference
        - and frankly I could've guessed they wouldn't be serverless
    - and I really should get in the habit of making runs as small as I can
    - the big thing is that the speedrunning frame isn't great for tasks involving runs like this
        - i guess e.g. minecraft speedruns have serial tasks and find ways to parallelize
        - there's less room to develop tech here since it's less consistent
        - but eh there's still stuff to try
    - so how am I gonna make good use of my runtimes
        - I should use alerts more for runs
        - Not much else occurs to me in the abstract, I'll write down anything that occurs concretely

## Split 4: test finetuning and forcing in a 2x2, start bigger ft jobs
- Target time: 1:30
- Start time: 8:36
- Notes:
    - Running 200 examples, og/ft100 X base/8k
        - results
            - og/base: 24%
            - ft100/base: 25%
            - og/force8k: 35%
            - ft100/force8k: 30%
        - ok so both ft and forcing improved performance
        - but I expected the ft model to gain more from forcing and actually the base model gained more
            - this is maybe because the base model is a r1 distill
        - now that I'm using ded ends I can use the actual correct model, lets do that next
    - Finetuning correct qwen model on 100 examples
    - Maybe it's worth testing literally their s1 model with my forcing implementation?
        - Need to go back to compute, Together doesn't support arbitrary HF models and inspect doesn't support forcing
            - Maybe it's worth a PR to inspect, prefilling is useful
        - Maybe worthwhile anyway for the XP
    - Aryan recommends a master doc to track results/open threads, I'll try it
- End time:
- Split time:
- Debrief notes:
    
5/23/25 2:30pm
Ok so I abandoned the splits format a bit since my availability was spotty, the training runs were long, and I was starting to shift more into debugging mode vs implementation mode
which makes less sense from a speedrunning perspective. Where am I at now.
My previous forcing implementation was busted, so I fixed it. I also fine-tuned Qwen32B on the full s1k dataset (1 epoch).
But now forcing makes the model worse instead of better. MATH accuracy drops from ~65% to ~30%.
Both the og and the ft versions produce ~1k chars unforced, and forcing them to 2k makes them give wrong answers and generally become incoherent.
Maybe this just means I really do need to run 5 epochs like the paper did? But I'm skeptical.
The s1 code removes stop tokens, which I shouldn't have to do since I'm just using the text
I'm gonna hop over to a different project for now, maybe more debrief to come later