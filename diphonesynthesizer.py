#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import simpleaudio
import argparse
import re
import numpy as np
import datetime
from string import Template
from nltk.corpus import cmudict

__author__ = "Kleber Noel"
__copyright__ = "Copyright 2019"
__license__ = "MIT License Agreement"
__version__ = "1.0.0"
__maintainer__ = "Kleber Noel"
__email__ = "klebnoel@gmail.com"
__status__ = "Protoype"

parser = argparse.ArgumentParser(
    description='A basic text-to-speech app that synthesises an input phrase using diphone unit selection.')
parser.add_argument('--diphones', default="./diphones", help="Folder containing diphone wavs")
parser.add_argument('--play', '-p', action="store_true", default=False, help="Play the output audio")
parser.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                    default=None)
parser.add_argument('phrase', nargs=1, help="The phrase to be synthesised")

parser.add_argument('--spell', '-s', action="store_true", default=False,
                    help="Spell the phrase instead of pronouncing it")
parser.add_argument('--crossfade', '-c', action="store_true", default=False,
					help="Enable slightly smoother concatenation by cross-fading between diphone units")
parser.add_argument('--volume', '-v', default=None, type=int,
                    help="An int between 0 and 100 representing the desired volume")

args = parser.parse_args()
cmu = cmudict.dict()

class Synth:
    """
    All synthesis procedures are dealt with here or in simpleaudio.
    """
    def __init__(self, wav_folder):
        self.diphones = {}
        self.get_wavs(wav_folder)
        self.wav_folder=wav_folder

    def get_wavs(self, wav_folder):
        """
        This function walks through the diphones directory and creates
        a dictionary from the present files iff they are not
        preceded by '._' (as in the mac computers pycharm creates
        alternate files preceded by this string) or followed by a
        '.wav' extension. The strings are updated the diphones
        dictionary with diphones as keys and files as values.

        :param wav_folder: diphones directory
        :return: self.diphones dict updated
        """
        for root, dirs, files in os.walk(wav_folder, topdown=False):
            for file in files:
                if file[0:2]!='._':
                    diphone=re.sub('(.wav)','',file)
                    self.diphones[diphone]=file

    def synthesize(self, diphonelist, crossfade=False):
        """
        This function checks for silence and appends diphones to a
        :param diphonelist: a list of diphones to be synthesized
        :param crossfade: argument passed through argpass that decides whether to crossfade diphones
        :return:
        """
        self.diphonesound = simpleaudio.Audio(rate=16000)
        self.diphone_wavdata_list=[]
        for key in diphonelist:
            self.silence_length=0
            try: # Which diphone file should be loaded?

                # Delete silence specification in string form (for now...)
                key_no_sil=re.sub('[24]','',key)

                # Create the string that can find the diphone file
                diphone_file = str(self.wav_folder + '/' + self.diphones[key_no_sil])

                # load it
                self.diphonesound.load(diphone_file)

                # put audio data into the list (diphone_wavdata_list is a list of arrays)
                self.diphone_wavdata_list.append(self.diphonesound.data)

            except Exception as e:
                strings=['Diphone {} not present in dictionary.'.format(e),'Backing off...',
                      'Searching for a diphone to fill in for {}'.format(e)]
                printdots(strings)

                # Attempt an emergency key search

                backupkey=self.emergency_diphone(key)

                # Create the string that can find the diphone file
                diphone_file = str(self.wav_folder + '/' + self.diphones[backupkey])

                # load it
                self.diphonesound.load(diphone_file)

                # put audio data into the list (diphone_wavdata_list is a list of arrays)
                self.diphone_wavdata_list.append(self.diphonesound.data)

            # investigate if a pau item had
            if key[-1] == '2':
                # 200ms of silence
                self.silence_length = 0.2

            if key[-1] == '4':
                # 400ms of silence
                self.silence_length = 0.4

            # append silence to the list if a value was added to variable self.silence_length during loop
            self.add_silence() if self.silence_length!=0 else None

        # reuse this from loading diphones, as the waveform settings/ internal objects will be correct
        self.new_object = self.diphonesound

        # join audio data chunks into one waveform
        self.crossfade() if args.crossfade else self.naively_concatenate()

        return self.new_object

    def naively_concatenate(self):
        self.new_object.data = np.concatenate(self.diphone_wavdata_list, axis=0) # Concatenate the diphone wavdata

    def add_silence(self):
        """
        Use the sampling rate, and length required
        to generate a numpy array for silence
        :return:
        """
        length=int(self.silence_length*self.diphonesound.rate)
        array = np.zeros(length, dtype=np.int16)
        self.diphone_wavdata_list.append(array)

    def crossfade(self, seconds=0.01):
        """
        This function concatenates the waveforms by using window
        length cross-fading.
        :param seconds:
        :return:
        """

        # initialise the windowlength
        windowlen=int(seconds*self.diphonesound.rate)

        flag=False
        diphones_array=np.empty

        # Begin by going through arrays in the saved list
        for array in self.diphone_wavdata_list:

            # take away the window length from the diphones array to create the resultant length
            resultantlen = int(len(array)-2*windowlen)

            # create a window of floating point number between 0 and 1, and
            # 1 and 0 of a length specified by the seconds argument
            windowarraystart  = np.linspace(0, 1, windowlen, dtype=np.float16)
            windowarrayfinish = np.linspace(1, 0, windowlen, dtype=np.float16)

            # create an array of ones between the resultant times
            ones=np.linspace(1,1,resultantlen, dtype=np.int)

            # merge the arrays created now of equal length
            windowfloatarray = np.concatenate((windowarraystart, ones, windowarrayfinish))

            # get the product of the windowed array (full floating points. [x...x], 0=>x=>1)
            # and the other diphone array (which is just the waveform array)
            windowedarray=windowfloatarray*array
            windowedarray=windowedarray.astype(dtype=np.int16)

            # Special case if statement:
            # Enter if statement if the first array has just been created,
            # and then equate it to diphones_array
            self.new_object.data=windowedarray
            if flag == False:

                flag = True
                diphones_array=windowedarray
                continue

            # calculate lengths of silence, from the length of all previous arrays
            # and the new array length to be added
            lensilencebefore = len(diphones_array)-windowlen
            lensilenceafter  = len(windowedarray)-windowlen

            # zeros before/after are two arrays of silence
            zerosbefore = np.zeros(lensilencebefore, dtype=np.int16)
            zerosafter  = np.zeros(lensilenceafter, dtype=np.int16)

            # create two arrays: one that is the result of the previous loops
            # and one that is the next array to be appended
            next = np.concatenate((zerosbefore, windowedarray), axis=0)
            prev = np.concatenate((diphones_array, zerosafter), axis=0)

            # add the arrays together to get a resultant waveform
            diphones_array=np.add(prev, next)

        # When outside the loop, concatenate the diphone wavdata
        self.new_object.data = diphones_array

    def emergency_diphone(self,lostkey):
        """
        Select an emergency diphone by using regex, this
        function will look through the dictionary's keys
        to find a key that is a near orthographic match
        to the lost key
        :param lostkey a key not in the dictionary
        :return: a new key to search
        """
        # midpoint of current diphone key
        midpoint=int()

        for i in range(len(lostkey)):

            if lostkey[i] == '-':
                midpoint=i
                break

        # keyfragment is the latter phone of the lost key (anything past '-')
        fragmentlatter=lostkey[midpoint+1:]
        fragmentformer=lostkey[:midpoint+1]

        printdots(['searching for emergency diphone'])

        # Go backwards from the end of the keyfragment
        for charindex in range(len(fragmentlatter.split())):

            # Initialise two variables to find the key
            star='*'
            ideal_key=str()
            try:
                # Two cases of latter key length:
                # 1: latter phone len == 2
                if len(fragmentlatter)==2:
                    s = Template('$a$b$c')
                    s.substitute(a=fragmentformer, b=fragmentlatter, c=star)
                    ideal_key = '{0}{1}{2}'.format(fragmentformer,fragmentlatter[charindex],star)

                # 2: latter phone len == 1
                elif len(fragmentlatter)==1:
                    s = Template('$a$b')
                    s.substitute(a=fragmentformer, b=star)
                    ideal_key = '{0}{1}'.format(fragmentformer, star)

                for k in self.diphones.keys():
                    if re.match(ideal_key,k):
                        strings=['{} found'.format(k)]
                        printdots(strings)
                        return k

            except:
                strings=['Error in key search','No emergency diphones were found.']
                printdots(strings)



class Utterance:
    """
    Front end: change raw input into a linguistic specification for synthesis.
    """
    def __init__(self, phrase):
        self.phrase=phrase
        self.diphonelist=list()
        self.dndict={'teens':{'19':"nineteen", '18':"eighteen", '17':"seventeen", '16':"sixteen", '15':"fifteen",
                    '14':"fourteen", '13':"thirteen", '12': "twelve", '11': "eleven"},

                    'digits':{'9':"nine", '8':"eight", '7':"seven", '6':"six", '5':"five", '4':"four", '3':"three",
                    '2':"two", '1':"one"},

                    'ordinals':{'1':"first", '2':"second", '3':"third", '4':"fourth", '5':"fifth", '6':"sixth",
                    '7':"seventh", '8':"eighth", '9':"ninth", '10':"tenth", '11':"eleventh",
                    '12':"twelfth", '13':"thirteenth", '14':"fourteenth", '15':"fifteenth", '16':"sixteenth",
                    '17':"seventeenth", '18':"eighteenth", '19':"nineteenth", '20':"twentieth",
                    '30':"thirtieth"},

                    'decimals':{'1':"ten",'2':"twenty",'3':"thirty",'4':"forty",'5':"fifty",'6':"sixty",
                    '7':"seventy",'8':"eighty",'9':"ninety",'0':"o"},

                    'hundreds':{'0':"hundred"},

                    'mil':{'00':'thousand'}
                    }

    def spell(self):
        """
        This splits the phrase into letters and separates
        using full-stops.
        """
        spelllist=[]
        tempphrase=' '.join(self.phrase)
        string='.,?!:; '
        for char in range(len(tempphrase)):

            if not tempphrase[char] in string :
                spelllist.append(str('{}.'.format(tempphrase[char])))

        self.phrase=spelllist

        del spelllist

    def letters(self, unk, i):
        """
        Resort to pronouncing the letters separately
        :param unk:
        :param i:
        :return:
        """
        pro=list()
        for j in unk:
            pro.append(cmu[j][i])
            print(j)
        return pro

    def unknownword(self, pron_attempt=[], unkword=None, i=0, flag=0):
        """
        Attempt to pronounce an unk (unknown word)
        by searching for words recursively!

        :param unkword: a unkword to pronounce
        :param flag: a flag that is used to indicate whether 3 recusion passes have been performed
        :return: cmu, letter for letter to pronounce the unkword
        """

        # begin indexing from the end of the word
        # (attempting to get the largest word possible)
        for index in range(len(unkword),0,-1):

            # create a variable that is the number of characters until the index
            chars=unkword[0:index]

            try:
                # If a pronunciation exists in the dictionary...
                if cmu[chars][i]:
                    # Add to flag, append pron_attempt and return to the function
                    # with variables updates.
                    flag += 1
                    pron_attempt.append(cmu[chars][i])
                    return self.unknownword(pron_attempt,unkword[index:],i, flag)
            # Naturally, there will be keyerrors attempting
            # to index the cmudict with nonsense
            except KeyError:
                continue

            # If the function has been called more that three times from inside the method
            # then break, returning with basic isolated letter pronunciation rules for the remainder
            # of the word.
            if flag > 2:

                return pron_attempt.append(self.letters(unkword[index - 1:], i))

        if len(unkword)==0:
            print(pron_attempt)
            resultantpronunciation = [phone for wordfound in pron_attempt for phone in wordfound]

            return resultantpronunciation


    def get_phone_seq(self):
        """
        Postcondition: Diphone sequence is generated

        How: Turns a phrase into a listed sequence of diphones
        ready to be read by the synthesizer by A. Preprocessing
        the utterance, B. I) Searching for tokens in the CMU lexicon
        B. II) reintroducing punctuation and C. Changing the
        phonelist into a diphone list.

        :return: self.diphonelist
        """
        pronunciation = []

        # Preprocess step 1a.: remove special chars, convert line to lower case, split line.
        self.clean()

        # Preprocess step 1b: preprocess i. dates and ii. numbers, iii. emphasis markers & update self.phrase
        self.preprocess_dates_numbers_emphasis()

        # Preprocess step 2: spell
        self.spell() if args.spell else None

        # Preprocess step 3a: store punctuation
        self.punctmarker=self.punctuation()

        # Preprocess step 3b: delete the remaining punctuation & update self.phrase
        self.delpunct()

        # Create a diphone word-marking list to keep track of words that become phones
        self.dp_word_em_marker=list()

        punctcount=0

        for wordindex in range(len(self.phrase)):

            word = self.phrase[wordindex]

            # Decide on a method later to choose an index depending on the word POS
            index_to_choose=0

            # Load a word:
            try:
                pronunciation.append(cmu[word][index_to_choose])

            except Exception as e:
                strings=['Error looking up {}'.format(e),
                         'Exception handler invoked to create a phone sequence']
                printdots(strings)

                unk=self.unknownword( [],word, index_to_choose, 0)
                print(pronunciation.append(unk))


            # Punctuation pause placement:
            try:
                # Reintroduce the punctuation markers into the utterance
                if wordindex == self.punctmarker[punctcount][0]:

                    # put the punctuation back into the list
                    pronunciation.append([self.punctmarker[punctcount][1]])

                    # add to the punctuation counter index
                    punctcount += 1
            except:
                continue

        # Return only the diphones list by joining the phones using '<phone>-<phone>'
        return self.diphones_from_cmu_seq(pronunciation)

    def clean(self):
        """
        Cleans the self.phrase string using regex
        and turns it into a list.

        :return: a lowercase list cleaned of punctuation
        """
        self.phrase=re.sub('[\^%$@)(><=+&\[\]`-]', '', self.phrase).lower().split()

    def preprocess_dates_numbers_emphasis(self):
        """
        This function normalizes dates, numbers and creates an emphasis
        marker, iff '{\w+}'

        :return: None
        """

        preprocess3 = []
        self.emphasis_marks = []

        for index in range(len(self.phrase)):
            self.paus_or_phone = self.phrase[index]

            # Check if the paus_or_phone is in date format
            if re.match('\d+/\d+(/\d+|)', self.paus_or_phone):

                try:
                    preprocess3.extend(self.process_date().split())
                except Exception as e:
                    strings=["Date error in preprocess_dates_numbers_emphasis...","{0}".format(e),
                    "Unable to process number","Discarding '{0}'".format(self.paus_or_phone)]
                    printdots(strings)
                    continue

            # Check if paus_or_phone is in number format (only whole integers can be read out)
            elif re.match('\d+', self.paus_or_phone):

                try:
                    preprocess3.extend(self.process_number().split())
                except Exception as e:
                    strings=["Number error in preprocess_dates_numbers_emphasis...","{0}".format(e),
                    "Unable to process number","Discarding '{0}'".format(self.paus_or_phone)]
                    printdots(strings)
                    continue

            # Check if the paus_or_phone is in emphasis format (emphasis addition still in development)
            elif re.match('[\{w+\}]', self.paus_or_phone):
                try:
                    re.sub('[\{\}]', '', self.paus_or_phone)
                    self.emphasis_marks.append(index)
                    preprocess3.append(re.sub('[\{\}]', '', self.paus_or_phone))

                except Exception as e:
                    strings = ["Emphasis {} brackets error in preprocess_dates_numbers_emphasis...", "{0}".format(e),
                               "Unable to process number", "Discarding '{0}'".format(self.paus_or_phone),
                               "Please be reminded that the next program version will have emphasis brackets"]
                    printdots(strings)
                    continue
            else:
                preprocess3.append(self.paus_or_phone)

        self.phrase=preprocess3

    def process_date(self):
        """
        process_date takes a paus_or_phone that has the format of a
        date and normalizes the digits using British conventions
        params:
        self.flag stores a True value if DD/MM is specified
        :return:
        """

        # Initialise two dictionaries: one for Ordinal days, one for year.
        format1 = "%d/%m/%Y"
        format2 = "%d/%m/%y"
        format3 = "%d/%m"

        if len(self.paus_or_phone) >= 8 and self.paus_or_phone[-5] == '/':
            self.flag=False
            self.process_date_try_except(format1)

        elif len(self.paus_or_phone) >= 6 and self.paus_or_phone[-3] == '/':
            self.flag=False
            self.process_date_try_except(format2)

        elif len(self.paus_or_phone) >= 3 and (self.paus_or_phone[-2] == '/' or self.paus_or_phone[-3] == '/'):
            self.flag=True
            self.process_date_try_except(format3)

        # Filter out the year, month and day of the date given
        year = self.dt.strftime("%Y") if not self.flag else None
        month = self.dt.strftime("%B").lower()
        day = self.dt.strftime("%d") if self.dt.strftime("%d") else None

        # Get the corresponding strings for the day and year

        # Day
        daystr=self.get_day_str(day)

        # Year
        ystr=self.get_year_str(year) if not self.flag else None

        # and return
        date_str_year = (' '.join([month, daystr, ystr])) if not self.flag else None

        date_str_no_year=(' '.join([month, daystr]))

        return date_str_no_year if self.flag else date_str_year

    def get_day_str(self, d):
        """
        This function processes a day numeric string
        and returns a linguistic string that represents the day.

        :param d: a two-char string of a numeric day (e.g. d='06')
        :return: a phrase denoting the input param d (e.g. dstr='sixth')
        """
        day_zero_to_nine=str()
        day_ten_to_teen =str()

        day_ten_to_teen = self.dndict['ordinals'][d] if d in self.dndict['ordinals'] else None

        bool=(day_ten_to_teen or d[-2]!=0)

        day_zero_to_nine = self.dndict['ordinals'][d[-1]] if self.in_d(d[-1],'ordinals') and not bool else None

        # If the date is not in the teens/ exception list of odd spelling then construct a string using indexing rules
        day_other=str()

        if not (day_zero_to_nine or day_ten_to_teen):
            dec = self.dndict['decimals'][d[0]]
            ord = self.dndict['ordinals'][d[1]]
            day_other = ('{} {}').format(dec,ord)

        day_to_word_list = [day_zero_to_nine, day_ten_to_teen, day_other]

        daystr = ' '.join(filter(None, day_to_word_list))

        return daystr

    def get_year_str(self, y):
        """
        This function takes a year (in string-numeric form from process_date
        and returns a linguistic string that represents the year.
        :param y: a four-char string of a numeric date (e.g. y='1999')
        :return: a phrase denoting the input param year (e.g. ystr='nineteen ninety nine'
        """

        # 1. Mid Digits:

        millenialexception = str()

        # Case a: generate 'thousand' for \d00\d. First check if date is millenial.
        mil_y = self.dndict['mil'][y[-3:-1]] if self.in_d(y[-3:-1],'mil') else None

        # 2. Last Digits

        # Case a: generate '-teen' at end of date. e.g. 12,13,14.
        # Check last two digits & assign iff they are teen.
        teen_y = self.dndict['teens'][y[-2:]] if self.in_d(y[-2:],'teens') else None

        # Case b: generate 'ten/-ty' at end of date. e.g. 20,40,60.
        # Check last two digits & assign str for these digits.
        decimal_zero_year = self.dndict['decimals'][y[-2]] if y[-1] == '0' and not y[-2:] == '00' else None

        # Case c: generate 'digit' at end of date iff end digit
        # is not equal to 0 (in the case of decimal_zero_year).
        last_digit = self.dndict['digits'][y[-1]] if self.in_d(y[-1],'digits') and not teen_y else None

        # Case d: Check the second to last digit in the date
        # and assign a str iff a digit has already been assigned.
        decimal_year = self.dndict['decimals'][y[-2]] if last_digit and not mil_y and self.in_d(y[-2],'decimals') else None

        # Case e: generate 'hundred'. Check last two digits.
        hundreds_year = self.dndict['hundreds'][y[-2]] if not mil_y and y[-2:] == '00' else None

        # 3. First Digits:

        # Case a: generate '-teen' at beg. of date. Check first two
        # digits & assign a str iff they are a teen.
        first_two_teens = self.dndict['teens'][y[:2]] if y[:2] in self.dndict['teens'] else None

        # Case b: generate 'ten/ -ty' at beg. of date iff first
        # two digits are of the form decimal-0.
        first_two_dec = self.dndict['decimals'][y[0]] if y[0] in self.dndict['decimals'] and not (first_two_teens or mil_y) else None

        # Case c: generate 'digit' at beg. of date iff mil_y=True and if
        # the first two digits are neither teens (19\d\d) nor decimals (20)
        first_digit = self.dndict['digits'][y[0]] if not (first_two_teens or first_two_dec) and mil_y else None

        # The following if statement adds the word 'and' between a millenial year and a digit
        # and sets the variables to none thereafter. e.g. (two thousand and six)
        if mil_y and last_digit:
            millenialexception = str('{0} and {1}').format(mil_y, last_digit)
            mil_y = None
            last_digit = None

        # Print these to create the linguistic date structure
        year_to_word_list = [first_digit, mil_y, first_two_teens, first_two_dec,
                             millenialexception, hundreds_year, decimal_zero_year, decimal_year, teen_y, last_digit]

        # The following line filters out any of the above values that are None type, creating the final string ystr.
        ystr = ' '.join(filter(None, year_to_word_list))

        return ystr

    def in_d(self, strdigits, dictkey):
        """
        Checks whether a str of digits is within the digit dictionary
        :param strdigits: a string of digits
        :param dictkey: a dictionary key
        :return: Truth value
        """
        return (strdigits in self.dndict[dictkey])

    def process_date_try_except(self,format):
        """
        :param self.paus_or_phone: a paus_or_phone being preprocessed
        :param format: a format for a datetime object
        :return: None, but updates self.dt
        """
        try:
            self.dt = datetime.datetime.strptime(self.paus_or_phone, format)
        except Exception as e:
            strings=["Failed to interpret {} as a string date".format(e),"nabandoning date processing"]
            printdots(strings)

    def process_number(self):
        """
        This function processes numbers
        from 1-9,999 in string format.

        :return: a normalized number string
        """

        number=self.paus_or_phone
        normalized_number=[]
        if len(number[-4:])==4:
            normalized_number.append(str('{0} thousand').format(self.dndict['digits'][number[-4]]))

        if len(number[-3:])==3 and self.dndict['digits'][number[-3]]:
            normalized_number.append(str('{0} hundred').format(self.dndict['digits'][number[-3]]))

        if len(number)>2 and not number[-2:]==00:
            normalized_number.append('and')

        if len(number[-2:])==2 and (self.dndict['digits'][number[-2]] and not number[-2:]==00):

            if number[-2]=='0':

                if number[-1] in self.dndict['digits'][number[-1]]:
                    normalized_number.append(str('{0}')).format(self.dndict['digits'][number[-1]])
                    return ' '.join(normalized_number)


            elif number[-2]=='1':
                normalized_number.append(str('{0}').format(self.dndict['teens'][number[-2:]]))
                return ' '.join(normalized_number)

            elif number[-2] in self.dndict['decimals']:
                normalized_number.append(str('{0}').format(self.dndict['decimals'][number[-2]]))

                if number[-1] in self.dndict['digits']:
                    normalized_number.append(str('{0}').format(self.dndict['digits'][number[-1]]))
                    return ' '.join(normalized_number)

        return ' '.join(normalized_number)

    def diphones_from_cmu_seq(self, pronunciation): # Initialise CMU sequence, add pauses, and turn into a diphone sequence
        """
        Initialise CMU sequence, add pauses, and
        turn into a diphone sequence
        :param pronunciation: the raw pronunciation from cmudict
        :return: a tuple of diphones
        """
        phonelist=[] # first, make a phonelist

        for cmupro in range(len(pronunciation)):
            for token in range(len(pronunciation[cmupro])):

                if pronunciation[cmupro][token] in '.:?!': # Some punctuation requires longer pauses
                    phonelist.append('pau4') # 400ms

                elif pronunciation[cmupro][token] in ',': # Other punctuation requires shorter pauses
                    phonelist.append('pau2') # 200ms

                else: # Most cases just require CMU substitution.
                    phonelist.append(re.sub('[0-9]', '', pronunciation[cmupro][token].lower()))

            if cmupro==len(pronunciation)-1 and phonelist[-1][-3:]!='pau': # Append pause
                phonelist.append('pau4')  # 400ms

        diphonelist=[]

        for phone in range(len(phonelist)-1): # This for loop creates the diphones using phonelist indicies

            diphonelist.append(str(phonelist[phone]+'-'+phonelist[phone+1]))

        del phonelist

        return tuple(diphonelist)

    def punctuation(self):
        """
        Takes the list self.phrase and checks each end index and as to
        whether the end char is a punctuation character.
        number which is used later in the pipeline to create a pause.
        :return: a tuple containing a punctuation marker and word index for pauses
        """
        punctuationplace=[]

        for i in range(len(self.phrase)):
            if self.phrase[i][-1] in '.,;:?!':
                punctuationplace.append((i,self.phrase[i][-1]))

        return tuple(punctuationplace)

    def delpunct(self):
        """
        deletes punctuation in the self.phrase list
        as one of the preprocessing steps in
        get_phone_seq()
        :return: None
        """
        q=[]
        for i in self.phrase:
            if args.spell:
                q.append(re.sub('[,;:?!]', '', i))
            else:
                q.append(re.sub('[.,;:?!]', '', i))

        self.phrase=q

def printdots(strings):
    """
    takes a list of strings and prints them nicely
    :param strings:
    :return: None (just prints a list)
    """
    for string in strings:
        print('{: ^150}'.format(string))

def welcome():
    """
    Welcome phrase is generated whenever the program is first run
    :return:
    """
    strings=['','','Welcome to your speech synthesis program','',
             'Please type in something you would like to be said',
             'or use the help -h key to investigate your options','',''
             ]
    printdots(strings)

if __name__ == "__main__":
    welcome()
    utt = Utterance(args.phrase[0])
    diphone_seq = utt.get_phone_seq()
    diphone_dict = Synth(wav_folder=args.diphones)
    dataobjectout=diphone_dict.synthesize(diphone_seq, args.crossfade)

    # Volume rescaling option
    if args.volume: dataobjectout.rescale(args.volume / 100)
    # Create the audio object
    out = simpleaudio.Audio(rate=16000)
    out.data = dataobjectout.data
    # Play option
    if args.play: out.play()

    # Save option
    if args.outfile:
        out.save(args.outfile)
        strings=['Your file have been saved as {}'.format(args.outfile)]
        printdots(strings)
