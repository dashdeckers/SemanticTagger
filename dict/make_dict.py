# File checks
import os.path
if os.path.isfile('output.py'):
    print("Please delete output.py, before creating a new one")
    exit()
if not os.path.isfile('input.txt'):
    print("Missing input.txt")
    exit()

# Read from tags.txt
with open('input.txt') as f:
    tags = [line.rstrip() for line in f]

# Write to tag_dict.py
with open('output.py', 'a') as f:
    f.write('tag_dict = {\n')
    for count, tag in enumerate(tags):
        f.write(f'    "{tag}": {count}')

        if count < (len(tags) - 1):
            f.write(',\n')
        else:
            f.write('\n')
    f.write('}')