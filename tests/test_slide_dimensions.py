import unittest, sys

from pathml import slide

testSlide = slide.Slide(
    '/media/gehrun01/archive-storage/BEST2/BEST2_POR_0001/BEST2_POR_0001_HE_1.svs',verbose=False)

#print(testSlide.slideProperties, sep=' ', end='n', file=sys.stdout, flush=False)
class TestSlideDimensions(unittest.TestCase):

    def test_slide_width(self):
        self.assertEqual(testSlide.slide.width, 111552,
                         "The slide width should be 111552px")
    def test_slide_height(self):
        self.assertEqual(testSlide.slide.height, 47515,
                         "The slide width should be 47515px")
    def test_slide_resolution(self):
        self.assertEqual(round(float(testSlide.slideProperties['openslide.mpp-x']),4), 0.2528,
                         "The MPP in x direction should be 0.2528")
        self.assertEqual(round(float(testSlide.slideProperties['openslide.mpp-y']),4), 0.2528,
                         "The MPP in y direction should be 0.2528")

if __name__ == '__main__':
    unittest.main()
