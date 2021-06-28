import sys
import unittest

from pathml import slide

testSlide = slide.Slide(
    '/media/gehrun01/archive-storage/BEST2/BEST2_POR_0001/BEST2_POR_0001_HE_1.svs', verbose=True)


class TestSlideLoader(unittest.TestCase):

    def test_slide_loader_type(self):
        self.assertEqual(testSlide.slideProperties['vips-loader'], 'openslideload',
                         "The file was not loaded using openslideload. Please check your pyvips library and dependencies")


if __name__ == '__main__':
    unittest.main()
