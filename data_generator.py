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


def filter_unwanted_characters(string):
    unwanted_chars = ['\'\'', '``', '\n']
    for unwanted_char in unwanted_chars:
        string = string.replace(unwanted_char, '')
    while '  ' in string:
        string = string.replace('  ', ' ')
    string = string.replace(' ,', ',')
    string = string.replace(';', ',')
    return string


def filter_to_reduce_vocabulary(string):
    output = []
    valid_chars_except_alpha_numeric = [' ', ',', '.', '$', '%']
    valid_chars_alphabetical = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for c in string:
        if c.isdigit():
            output.append('@')
        if c in valid_chars_alphabetical:
            output.append(c)
        if c in valid_chars_except_alpha_numeric:
            output.append(c)
    return ''.join(output).lower()


# Cross entropy = 1.1835 on 1e3.
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
                new_str = ''.join([v for v in new_lines if not v.startswith('--') and '@' not in v]).strip()
                new_str = filter_unwanted_characters(new_str)

                if SHRINK_VOCABULARY_SIZE:
                    new_str = filter_to_reduce_vocabulary(new_str)

                buffer += new_str
    print('data length =', len(buffer))
    return buffer


if __name__ == '__main__':
    read()
