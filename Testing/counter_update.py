import os

# Set the directory where your txt files are located
directory = './Testing/'

# Iterate over each file from counter1.txt to counter18.txt
for i in range(1, 2):
    filename = f'counter{i}.txt'
    filepath = os.path.join(directory, filename)
    
    # Write '1' to the file
    with open(filepath, 'w') as file:
        file.write('1')
    
    print(f'Updated {filename}')

print('Counter updated')
