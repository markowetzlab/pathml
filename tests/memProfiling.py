import pyvips

for i in range(10000):
    print(f"loop {i} ..")
    im = pyvips.Image.new_from_file(sys.argv[1], access="sequential")
    im.shrink(16, 16).write_to_file(sys.argv[2])
