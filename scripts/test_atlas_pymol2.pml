from pymol import cmd

cmd.load("/sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas/proteins/1ez3_B/1ez3_B.pdb", "prot")

print("After load - atoms:", cmd.count_atoms("prot"))
print("After load - states:", cmd.count_states("prot"))

# Check chain IDs
stored_chains = []
cmd.iterate("prot and name CA", "stored_chains.append(chain)", space={"stored_chains": stored_chains})
unique_chains = set(stored_chains)
print("Chain IDs found:", unique_chains)

# Remove water/non-protein
cmd.remove("resn HOH")
cmd.remove("not polymer.protein")
print("After cleanup - atoms:", cmd.count_atoms("prot"))

# Try split_states
n_states = cmd.count_states("prot")
print("Number of states:", n_states)
if n_states > 1:
    cmd.split_states("prot", 1, 1)
    print("After split - prot_0001 atoms:", cmd.count_atoms("prot_0001"))
    cmd.delete("prot")
    cmd.set_name("prot_0001", "prot")
else:
    print("Single state, no split needed")

print("Final atom count:", cmd.count_atoms("prot"))

# Try rendering
cmd.show("cartoon", "prot")
cmd.orient("prot")

# Test: alter b-factors
cmd.alter("prot", "b=50")
cmd.spectrum("b", "blue_white_red", "prot")
cmd.ray(800, 600)
cmd.png("/tmp/test_atlas2.png", width=800, height=600)
print("Saved test image")

cmd.quit()
