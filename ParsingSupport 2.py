# comment

import re

def find_pattern_with_last_word(tarfile,pattern):

	'''
		fine pattern in a text
	'''

	#print(tarfile,pattern)
	total_lnumber = 0

	# Compile the regular expression pattern
	pattern = re.compile(pattern)

	with open(tarfile,'r') as file:
		lines = file.readlines()
		total_lnumber = len(lines)
	matches = [] 

	for lnumber, line in enumerate(lines):
		#match = re.search(pattern,line)
		match = pattern.search(line)

		if match:
			last_word = line.strip().split()[-1]
			matches.append([lnumber+1,last_word])

	return total_lnumber,matches

def get_lines(tarfile,line_start,line_end):

	'''
		line arugments human indexing convention
	'''
	with open(tarfile,'r') as file:
		lines = file.readlines()

	return lines[line_start-1:line_end-1]

def time_to_seconds(time_str):
	parts = time_str.split(':')
	hours = int(parts[0])
	minutes = int(parts[1])
	seconds = float(parts[2])
	total_seconds = hours * 3600 + minutes * 60 + seconds
	return total_seconds

if __name__ == '__main__':

	#/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_1/FHIaims.out 'Begin self-consistency iteration #'
	file_path = "/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_1/FHIaims.out"
	pattern = 'Begin self-consistency iteration #'

	matches = find_pattern_with_last_word(file_path,pattern)
	print(matches)
	holder = get_lines(file_path,1,15)

	for item in holder:
		print(item)
