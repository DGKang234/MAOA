from AppOutputExtractor.FHIaims.FHIaimsOutputExtractor import extractor


app_output = './aims.out'

extractor = extractor()
extractor.set_output_filepath(app_output)

species = extractor.get_species
no_atoms = extractor.get_no_atoms
geometries = extractor.get_geometries
order = extractor.get_atom_order
forces = extractor.get_forces
print(forces)
