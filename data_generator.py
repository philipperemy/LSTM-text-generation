import os

BASE_DIR = 'financial-news-dataset/'
BLOOMBERG_DATA_DIR = os.path.join(BASE_DIR, '20061020_20131126_bloomberg_news')
REUTERS_DATA_DIR = os.path.join('ReutersNews106521')


def get_filename():
    for root, dirs, files in os.walk(BLOOMBERG_DATA_DIR):
        for file in files:
            yield os.path.join(root, file)
    for root, dirs, files in os.walk(REUTERS_DATA_DIR):
        for file in files:
            yield os.path.join(root, file)

# Cross entropy = 1.1835 on 1e3.
def read(num_filenames=int(3e3)):
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
                new_str = new_str.replace('\'\'', '').replace('``', '').replace('  ', ' ').replace('\n', ' ').replace(
                    ' ,', ',').replace('  ', ' ').replace('  ', ' ')
                buffer += new_str
    print('data length =', len(buffer))
    return buffer


if __name__ == '__main__':
    read()

