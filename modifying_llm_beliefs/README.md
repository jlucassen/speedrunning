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
- Stop time: 
- Split time: 
- Debrief notes: