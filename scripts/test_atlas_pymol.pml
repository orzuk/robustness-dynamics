from pymol import cmd
cmd.load("/sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas/proteins/1ez3_B/1ez3_B.pdb", "protein")
cmd.remove("not polymer.protein")
print("Atom count:", cmd.count_atoms("protein"))
cmd.show("cartoon", "protein")
cmd.orient("protein")
cmd.ray(800, 600)
cmd.png("/tmp/test_atlas.png")
cmd.quit()
