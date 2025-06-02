# Modifying LLM Beliefs (With Synthetic Document Finetuning)
- Less open-ended research, more implementation

## Split 1: Synthetic document construction pipeline
- Target time: :45
- Start time: 7:25
- Notes:
- Stop time: 8:10
- Split time: :45
- Debrief notes:
    - whooo right in under the wire
    - neat! very simple+concrete, ez.
    - next thing is doing the finetuning, which should also be quick
        - one of these days I should do finetuning on a GPU instead of via API
        - or should I? Is that something people still do in 2025?
    - whoa what Claude finetuning is only via Amazon bedrock? 

## Split 2: Finetune Llama 3.3 70B
- Target time: 1:30
- Start time: 7:00
- Notes:
    - GPU finetuning worth practicing, trying out lambda instead of runpod
        - ok so I need like 150GB VRAM for inference
        - and ~3x that for full finetuning, but how much does LoRA need?
            - ~1.3x, nice
    - Gonna do some testing with 8b and see how pricey it is
    - Ok training is running now, dunno how long I've spent debugging, current workflow:
        - connect to runpod server
        - cd to /dev/shm, clone
        - chmod and run runpod_hf.sh
        - download target unsloth model from hf, using hf
        - move downloaded model to /unsloth/modelname
        - install unzip, unzip documents.zip
        - finetune.py
- Stop time: 
- Split time: 
- Debrief notes:

## Split 3: Implement 4 Belief Measurements
- Target time: 2:00
- Start time: 9:47
- Notes:
    - the plan? make an LLM pipeline that converts lists of false facts into HF datasets of the 4 formats the blog post describes
    - took 24m to make the bit that creates the mcq knowledge questions
        - ehhh vaguely on pace, should be v quick to make the evaluator, but the other will be more complicated. need to pick it up
    - putting this on pause to head to the farmers market
        - also just fixed the septerra motors facts bc many of them don't make sense in isolation
        - heading out 10:10
        - back at 11:30
    - lunch
        - out at 2:30
        - back 3:15
- Stop time: 4:00
- Split time: 4:08
- Debrief notes:
    - man did that take a long time
    - implementing the basic measurement methods didn't take too long I think
    - it's not like anything really derailed me on this one and made it take a long time
    - might debrief more later, surely I could've done this faster but no big speedup opportunities jump out at me, maybe it's just a matter of better mechanics and fewer minor bugs?
    - took some time to fix the synthetic document generation pipeline to do single facts, more analogous to the blog post

## Split 4: Compare two different models pre-/post-finetune
- Target time: 1:00
- Start time: 2:15
- Notes:
    - This is basically to make sure my finetuning pipeline is working smoothly before I go for 70B
    - finetuning script fixed
    - 16b saves work out of the box with vllm
    - next just need to update my evaluation question generation pipeline