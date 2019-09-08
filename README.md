# diphonesynthesizer
A simple synthesizer for that uses diphone segments.

Given a sub folder of diphone segments, e.g. with filenames like 'ah-m.wav',
the script should be able to synthesize sentences using linguistic knowledge 
encoded in functions the program excecutes and the pronunciation dictionary 
it uses.

Additionally, the program can attempt to synthesize words OOV, by 
recursively searching dictionary enteries available in cmudict
using sets of substrings until a solution is found.
