import ParsingSupport

file_path = "/Users/woongkyujee/Desktop/Python/FHI22_samples/runs/run_1/FHIaims.out"
pattern = 'Begin self-consistency iteration #'

lnumber, matches = ParsingSupport.find_pattern_with_last_word(file_path,pattern)
print(lnumber, matches)
