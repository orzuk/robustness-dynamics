from pymol import cmd
cmd.fragment("ala")
cmd.ray(800, 600)
cmd.png("/tmp/test_pymol.png")
cmd.quit()
