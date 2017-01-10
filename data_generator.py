import os

BASE_DIR = 'financial-news-dataset/'
BLOOMBERG_DATA_DIR = os.path.join(BASE_DIR, '20061020_20131126_bloomberg_news')
REUTERS_DATA_DIR = os.path.join('ReutersNews106521')

SHRINK_VOCABULARY_SIZE = True


# @ is the encoding for digits.


def get_filename():
    for root, dirs, files in os.walk(BLOOMBERG_DATA_DIR):
        for file in files:
            yield os.path.join(root, file)
    for root, dirs, files in os.walk(REUTERS_DATA_DIR):
        for file in files:
            yield os.path.join(root, file)


def filter_unwanted_characters(s):
    unwanted_chars = ['\'\'', '``', '\n']
    for unwanted_char in unwanted_chars:
        s = s.replace(unwanted_char, ' ')
    while '  ' in s:
        s = s.replace('  ', ' ')
    s = s.replace(' ,', ',')
    s = s.replace(';', ',')
    end_tag = 'To contact the reporter'
    if end_tag in s:
        s = s[:s.index(end_tag)]
    return s


def filter_to_reduce_vocabulary(string):
    string = string.lower()
    output = []
    valid_chars_except_alpha_numeric = [' ', ',', '.', '$', '%', '\'', '-']
    valid_chars_alphabetical = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for c in string:
        if c.isdigit():
            output.append('@')
        if c in valid_chars_alphabetical:
            output.append(c)
        if c in valid_chars_except_alpha_numeric:
            output.append(c)
    return ''.join(output)


def read(num_filenames=int(6e3)):
    buffer = ''
    for i, file in enumerate(get_filename()):
        filename = file.split('/')[-1]
        print(i, filename)
        if i > num_filenames:
            break
        if '-' in filename:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                new_lines = f.readlines()
                st = max([t[0] for t in enumerate(new_lines) if t[1].startswith('--')]) + 1
                new_str = ''.join([v for v in new_lines[st:] if not v.startswith('--') and '@' not in v]).strip()
                new_str = filter_unwanted_characters(new_str)
                if SHRINK_VOCABULARY_SIZE:
                    new_str = filter_to_reduce_vocabulary(new_str)
                print(new_str)
                buffer += new_str
    print('data length =', len(buffer))
    return buffer


if __name__ == '__main__':
    read()
