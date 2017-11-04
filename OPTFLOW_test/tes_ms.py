import cv2
import pymeanshift as pms

original_image = cv2.imread("gamma2.png")

my_segmenter = pms.Segmenter()

my_segmenter.spatial_radius = 6
my_segmenter.range_radius = 1
my_segmenter.min_density = 100

(segmented_image, labels_image, number_regions) = my_segmenter(original_image)
cv2.imwrite('gammaout2.png', segmented_image)