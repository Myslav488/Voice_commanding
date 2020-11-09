# Voice_commanding
For greater purpose.

Hi,
This my project based on Raspberry PI.

It's purpose is to recognize voice commands and make proper decision.
Signals gathered by microphone are send to RPi by USB Sound Card, 
where they are digitised. Signal is procesed and then algorithm 
detects parts of it where there may be a voice signal. After cutting 
proper piece of signal it is compared with Hidden Markov Model built
on base of pre-recorded comands. When the correct command is detected,
microprocessor gives the decidions for motors to make a move. Like for
instance if You say 'move forward' the motors will spin on and 
whole structure will move forward. 

I will give better explanation soon. 
PG
