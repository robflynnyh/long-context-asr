import re, number_conversion # https://github.com/robflynnyh/number_conversion 

def post_process(text): # replace with space
    text = re.sub('(\d+)%', r'\1 %', text).replace(' %', ' percent') # 100% -> 100 percent
    text = re.sub(r"(?<=\d)\.(?=\d)", " point ", text) # 1.5 -> 1 point 5
    text = re.sub(r'(?<=\d),(?=\d)', '', text) # 1,000 -> 1000
    text = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", text) # $1000 -> 1000 dollars
    text = re.sub(r"\£(\d+(?:\.\d+)?)", r"\1 pounds", text) # £1000 -> 1000 pounds
    text = re.sub(r"\€(\d+(?:\.\d+)?)", r"\1 euros", text) # €1000 -> 1000 euros
    # punctuation = everything except "'" which is used in contractions
    text = re.sub(r"[^\w\d'\s]+",' ',text)
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    text  = text.replace('u238', 'u two thirty eight')
    text = text.replace('u235', 'u two thirty five')
    text = text.replace('1900s', 'nineteen hundreds')
    text = text.replace('1910s', 'nineteen tens')
    text = text.replace('1920s', 'nineteen twenties')
    text = text.replace('1930s', 'nineteen thirties')
    text = text.replace('1940s', 'nineteen forties')
    text = text.replace('1950s', 'nineteen fifties')
    text = text.replace('1960s', 'nineteen sixties')
    text = text.replace('1970s', 'nineteen seventies')
    text = text.replace('1980s', 'nineteen eighties')
    text = text.replace('1990s', 'nineteen nineties')
    text = text.replace('2000s', 'two thousands')
    text = text.replace('2010s', 'two thousands tens')
    text = text.replace('2020s', 'two thousands twenties')
    text = text.replace('1st', 'first')
    text = text.replace('2nd', 'second')
    text = text.replace('3rd', 'third')
    text = text.replace('4th', 'fourth')
    text = text.replace('5th', 'fifth')
    text = text.replace('6th', 'sixth')
    text = text.replace('7th', 'seventh')
    text = text.replace('8th', 'eighth')
    text = text.replace('9th', 'ninth')
    text = text.replace('10th', 'tenth')
    text = text.replace('11th', 'eleventh')
    text = text.replace('12th', 'twelfth')
    text = text.replace('13th', 'thirteenth')
    text = text.replace('14th', 'fourteenth')
    text = text.replace('15th', 'fifteenth')
    text = text.replace('16th', 'sixteenth')
    text = text.replace('17th', 'seventeenth')
    text = text.replace('18th', 'eighteenth')
    text = text.replace('19th', 'nineteenth')
    text = text.replace('20th', 'twentieth')
    text = text.replace('21st', 'twenty first')
    text = text.replace('22nd', 'twenty second')
    text = text.replace('23rd', 'twenty third')
    text = text.replace('24th', 'twenty fourth')
    text = text.replace('25th', 'twenty fifth')
    text = text.replace('26th', 'twenty sixth')
    text = text.replace('27th', 'twenty seventh')
    text = text.replace('28th', 'twenty eighth')
    text = text.replace('29th', 'twenty ninth')
    text = text.replace('30th', 'thirtieth')
    text = text.replace('31st', 'thirty first')

    text = re.sub(r"(?<=\w)'(?=\s|$)", "", text) # remove trailing apostrophe i.e persons' -> persons but not person's
    text = number_conversion.convert_doc(text)
    text = text.replace('-', ' ')
    text = text.replace(' dr ', ' doctor ')
    text = text.replace(',', '')

    return text