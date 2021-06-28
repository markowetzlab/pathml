import sys
import pyvips

image = pyvips.Image.new_from_file(sys.argv[1],level=2)

patch_size = 512
n_across = image.width // patch_size
n_down = image.height // patch_size
x_max = n_across - 1
y_max = n_down - 1

n_patches = 0
for y in range(0, n_down):
    print("row {} ...".format(y))
    for x in range(0, n_across):
        patch = image.crop(x * patch_size, y * patch_size,
                           patch_size, patch_size)
        patch_f = patch.fliphor()

        patch0 = patch.write_to_memory()
        patch1 = patch.rot90().write_to_memory()
        patch2 = patch.rot180().write_to_memory()
        patch3 = patch.rot270().write_to_memory()

        patch4 = patch_f.write_to_memory()
        patch5 = patch_f.rot90().write_to_memory()
        patch6 = patch_f.rot180().write_to_memory()
        patch7 = patch_f.rot270().write_to_memory()

        n_patches += 8

print("{} patches generated".format(n_patches))
