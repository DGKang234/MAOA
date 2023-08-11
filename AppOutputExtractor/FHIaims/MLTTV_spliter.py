from ase import io 

def split_xyz_file(input_file, train_file, valid_file, test_file):
    with open(input_file, 'r') as infile:
        train_out = open(train_file, 'w')
        valid_out = open(valid_file, 'w')
        test_out = open(test_file, 'w')

        while True:
            # Read the header line containing the number of atoms
            header = infile.readline()
            if not header:
                break  # End of file

            num_atoms = int(header.strip())
            block_lines = [infile.readline() for _ in range(num_atoms + 1)]

            # Determine which file to write to based on the current index
            i = infile.tell()  # Get current position in file
            if i % 5 < 3:
                output_file = train_out
            elif i % 5 == 3:
                output_file = valid_out
            else:
                output_file = test_out

            # Write the block to the chosen file
            output_file.write(header)
            output_file.writelines(block_lines)

        train_out.close()
        valid_out.close()
        test_out.close()

# Usage:
split_xyz_file('Training_set.xyz', 'Training_set_test.xyz', 'Validation_set_test.xyz', 'Testing_set_test.xyz')

