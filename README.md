# jpeg_base_generator
This script create JPEG bases with RAWs bases (like ALASKA, BOSS, ...)
It's possible to have images in gray scale but the most important is the possibility to split all images and to choose the size of images.
You control all developments applied on your images.

For example, you create the base with JPEG 1024x1024 and want to split in 16 images 256x256, it's possible.

To count the number of files during the creation of the base, you can use this command line in the script directory:
clear && find JPEG_Bases/Real_Base_3_JPG_1024x1024_QF75/ -name "*.jpg" | wc -l
