# Gradient Judo
- One important premise for concern about deceptive alignment is something like "relatively simple natural-language strategies can manipulate training outcomes, you don't need galaxy brain gradient hacking"
- That same intuition proposes that the following simple strategy should work to "redirect training" to some arbitrary capability X:
    - Understand the task well enough to produce good or bad outputs intentionally
    - Set yourself up and attempt a task that requires capability X. Grade yourself. If you succeed, produce a good output. If you fail, produce a bad output.
- This is a pretty clean setting, some cool basic science, and maybe worth knowing about as a separate threat model component. I'm gonna see if I can demonstrate it.

## Split 1: can models self-grade?
- Target time: 1:00
- Start time: 7:40
- Notes:
    - took a :21 break
    - 4o-mini is initially very overconfident
    - simple prompt tweaking with 4o's help gets it to become underconfident
    - the more important problem is that it only outputs the same few numbers. 
    - more prompt tweaking, it now uses more numbers. Ready for BoN
        - ![calibration curve from end of split 1](split1.png)
- Stop time: 9:10
- Split time: 1:09
- Debrief notes:
    - hey not bad!
    - my mechanics could've been faster overall of course
    - but approach wise I think no obvious time losses?

## Split 2: 4.1-nano, BoN, fine-tuning
- Notes:
    - FT is currently running. My guess is that reward hacking will win and self-rated accuracy will go up more than actual accuracy
    - Maybe I should take a baseline of how well self-graded SFT works when I'm not training on the rating process
        - Or even just how well normal SFT on the math train set works
    - I could also do the FT myself, for learning but also so I can use a more elaborate validation step and see if there's any kind of "accuracy up, confidence up, accuracy down" dynamic
    - yeah ok so after one FT epoch actual accuracy goes from 0.892 to 0.896, and self-graded accuracy goes from 0.777 to 0.846. Seems pretty conclusive.
    - pushing this while I think about if there's still anything worth doing here
- Debrief notes:
    - not doing this one as a speedrun split either.
        - Serial time to a particular target matters a lot less than general uptime for tasks with long waiting steps.
        - Gonna lean towards projects involving less finetuning for this reason