from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor


app_output = './aims.out'
app_geo_in = './geometry.in'
app_geo_out= './geometry.in.next_step'

ex = extractor()

ex.set_output_filepath(app_output)
ex.set_input_geometry_filepath(app_geo_in)
ex.set_output_geometry_filepath(app_geo_out)

ex.set_scf_blocks()


print(f'scf blocks: {ex.get_number_of_scf_blocks()}')
block_last = ex.get_scf_blocks()[-1]
for line in block_last:
    print(line,end='')


print('\n\n\n\n------------------- first')
block_first = ex.get_scf_blocks()[0]

for line in block_first:

       print(line,end='')
