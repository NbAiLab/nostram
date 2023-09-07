symbols = {'%': ' posto', '@': 'et', '¾': 'tri četvrtine', '¼': 'četvrtina', '½': 'pola', '&': 'end', '°': 'stupnja',
           '€': 'euro', 'π': 'pi'}
sym_ignore = {'!', '?', ',', '.', '..', '...', '....', "'", '(', ')', '+', '-', ':', '–', '“', '„', '"', '/', '\\', '#',
              '=', '¬', '´', '¸', '’', '—', ';','<','>'}

abbreviations = {
    'HNB': 'haenbe',
    'Sv': 'sveti',
    'RTV': 'erteve',
    'cm': 'centimetar',
    'PDV': 'pedeve',
    'BBC': 'bibisi',
    'd': 'de',
    'HSU': 'haesu',
    'BiH': 'beiha',
    'HRT': 'haerte',
    'SDSS': 'esdeeses',
    'tzv': 'takozvani',
    # 's': 'es',#TODO
    'kg': 'kilogram',
    'k': 'ka',
    'HSS': 'haeses',
    'HNS': 'haenes',
    'tj': 'tojest',
    'br': 'broj',
    'HTV': 'hateve',
    'RTL': 'erteel',
    'BDP': 'bedepe',
    'HZZO': 'hazezeo',
    'HDZ': 'hadeze',
    'Npr': 'naprimjer',
    'itd': 'itede',
    'TV': 'teve',
    'HAVC': 'havece',
    'b': 'be',
    'vd': 'vede',
    'PFU': 'peefu',
    'VD': 'vede',
    'tv': 'teve',
    'hr': 'haer',
    'SDP': 'esdepe',
    'npr': 'naprimjer',
    'rtv': 'erteve',
    'c': 'ce',
    'MAX': 'maks',
    'S': 'es',
    'v': 've',
    'kn': 'kuna',
    'RH': 'erha'
}

unit2str = {
    0: 'nula',
    1: 'jedan',
    2: 'dva',
    3: 'tri',
    4: 'četiri',
    5: 'pet',
    6: 'šest',
    7: 'sedam',
    8: 'osam',
    9: 'devet'
}

thunit2str = {
    1: 'jedna',
    2: 'dvije',
    3: 'tri',
    4: 'četiri',
    5: 'pet',
    6: 'šest',
    7: 'sedam',
    8: 'osam',
    9: 'devet'
}

teen2str = {
    10: 'deset',
    11: 'jedanaest',
    12: 'dvanaest',
    13: 'trinaest',
    14: 'četrnaest',
    15: 'petnaest',
    16: 'šesnaest',
    17: 'sedamnaest',
    18: 'osamnaest',
    19: 'devetnaest'
}

ten2str = {
    20: 'dvadeset',
    30: 'trideset',
    40: 'četrdeset',
    50: 'pedeset',
    60: 'šezdeset',
    70: 'sedamdeset',
    80: 'osamdeset',
    90: 'devedeset'
}

hundred2str = {
    100: 'sto',
    200: 'dvijesto',
    300: 'tristo',
    400: 'četristo',
    500: 'petsto',
    600: 'šesto',
    700: 'sedamsto',
    800: 'osamsto',
    900: 'devetsto'
}


def number_to_text(number: int) -> str:
    if number == 0:
        return 'nula'

    assert number > 0

    ret = []

    if number > 999999999999:
        for c in str(number):
            ret.append(unit2str[int(c)])
        return ' '.join(ret)

    bilions = number // 1000000000
    number -= bilions * 1000000000
    milions = number // 1000000
    number -= milions * 1000000
    thousands = number // 1000
    number -= thousands * 1000
    hundreds = number // 100
    number -= hundreds * 100
    tens = number // 10
    units = number % 10

    if bilions:
        hundredbilions = bilions // 100
        bilions -= hundredbilions * 100
        tenbilions = bilions // 10
        bilions = bilions % 10

        if hundredbilions > 0:
            ret.append(hundred2str[hundredbilions * 100])
        if tenbilions > 1:
            ret.append(ten2str[tenbilions * 10])
            if bilions > 0:
                ret.append(thunit2str[bilions])
        elif tenbilions == 1:
            ret.append(teen2str[10 + bilions])
        elif bilions > 0:
            ret.append(thunit2str[bilions])
        if bilions == 1:
            ret.append('milijarda')
        elif 2 >= bilions >= 4:
            ret.append('milijarde')
        else:
            ret.append('milijardi')

    if milions:
        hundredmilions = milions // 100
        milions -= hundredmilions * 100
        tenmilions = milions // 10
        milions = milions % 10

        if hundredmilions > 0:
            ret.append(hundred2str[hundredmilions * 100])
        if tenmilions > 1:
            ret.append(ten2str[tenmilions * 10])
            if milions > 0:
                ret.append(unit2str[milions])
        elif tenmilions == 1:
            ret.append(teen2str[10 + milions])
        elif milions > 0:
            ret.append(unit2str[milions])
        if milions == 1:
            ret.append('milijun')
        else:
            ret.append('milijuna')

    if thousands:
        hundredtousands = thousands // 100
        thousands -= hundredtousands * 100
        tenthousands = thousands // 10
        thousands = thousands % 10

        if hundredtousands > 0:
            ret.append(hundred2str[hundredtousands * 100])
        if tenthousands > 1:
            ret.append(ten2str[tenthousands * 10])
            if thousands > 0:
                ret.append(thunit2str[thousands])
        elif tenthousands == 1:
            ret.append(teen2str[10 + thousands])
        elif thousands > 0:
            ret.append(thunit2str[thousands])
        if thousands == 2 or thousands == 3:
            ret.append('tisuće')
        else:
            ret.append('tisuća')

    if hundreds > 0:
        ret.append(hundred2str[hundreds * 100])
    if tens > 1:
        ret.append(ten2str[tens * 10])
        if units > 0:
            ret.append(unit2str[units])
    elif tens == 1:
        ret.append(teen2str[10 + units])
    elif units > 0:
        ret.append(unit2str[units])

    return ' '.join(ret)
