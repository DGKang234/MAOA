
import re

def find_pattern_in_file(file_path, pattern):
    with open(file_path) as f:
        # Read the file line by line
        lines = f.readlines()

    # Use regular expression to search for the pattern
    regex = re.compile(pattern)
    matching_lines = [i+1 for i, line in enumerate(lines) if regex.search(line)]

    # Return the line numbers where the pattern is found
    return matching_lines

def find_pattern_with_last_word(filename, pattern):

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    matches = []

    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            last_word = line.strip().split()[-1]
            matches.append((i+1, last_word))
    
    return matches


if __name__ == '__main__':

	file_path = "/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_1/FHIaims.out"
	pattern = r"SCF"  # The pattern you want to search for
	pattern = 'Begin self-consistency iteration #'

	#matching_lines = find_pattern_in_file(file_path, pattern)
	#print("Matching line numbers:", matching_lines)


	print('----- test 2 -----')

	matches = find_pattern_with_last_word(file_path,pattern)
	print(matches)
