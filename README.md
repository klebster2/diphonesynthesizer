# diphonesynthesizer
A simple synthesizer for that uses diphone segments.

Given a sub folder of diphone segments, e.g. with filenames that 
give a phonological specification 'ah-m.wav', the script should be 
able to synthesize sentences using linguistic knowledge encoded in 
the functions and the cmu pronunciation dictionary the program uses.

Additionally, the program can attempt to synthesize words OOV, by 
recursively searching dictionary enteries available in cmudict using 
truncated substrings until a solution is found.
