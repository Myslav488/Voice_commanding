# Voice_commanding

This is my master thesis project.

It's aim is to make system on RPi 4B for contoiling medical equipment by voice commands.

Although it recognize some commands yet, the work is still in progress and i am upgrading it.

Done:
> recording and loading audio signal from external USB sound card
> procesing data
> detecting and extracting voice activity
> getting HFCC features of extracted command
> learn commands by a Hidden Markov Model from base of records
> recognize voice record by the Hiden Markov Model's algorithm 
> current efficiency of recognition: ~20%

To do:
> upgrade voice activity detection and HMM to get best possible score
> attach system to physical vehicle (executive set)

The commands and most variables in a code are typed in polish, as it is language of my thesis. 
