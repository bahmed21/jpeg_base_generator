import numpy as np
import os
import shutil
import multiprocessing
import tifffile
import datetime
import time
from PIL import Image
from joblib import Parallel, delayed
from hashlib import md5
from pathlib import Path
from subprocess import call

# import image_conversion_fun as imProc
import image_conversion_fun_v2 as imProc
from random_dev import devRandomGenerator
from fix_dev import devFixGenerator

# Note that this first variable is used to allows several development, with possibly various settings, with names
# that can be easily identified Switch indicating whether an uncompressed version (tiff) of the developed images
# should be kept. Useful for steganography / steganalysis / forensics in spatial domain
keepUncompressed = False

# Name of the directory that will contain developed JPEG images:
baseName = "Real_Base"
# Fix the development for all images in RAWs DB or make a random development
bool_random_dev = False
# To divide the initial image in 16 small images
# /!\ The multi crop is disable if th development is random
bool_multicrop = True
# If we want all images in gray, active the following boolean
bool_grayscale = True

# Remove all files or not
bool_remove_beginning = True
remove_dir = ["out_dir", "out_dir_tif", "tmp_dir", "profile_used_dir", "out_dir_multisplit"]

# Create a text file to store every X3F images
with open('X3F_images.txt', 'w') as f:
    f.write("All X3F images are list here:\n")
    f.close()

# Complete path where the script is run
real_path = Path(os.path.dirname(os.path.realpath(__file__)))
# Here we start defining the most important variable "config_path" which defines ALL directory used
# we start with the main root directory, in which everything will be output:
config_path = dict(root="JPEG_Bases")
# config_path = dict(root="Dell_Bases" + "/" + baseName)
# where RAW images should be
raw_folder_path_parent = os.path.join(real_path.parent.parent, "RAW_bases")
# raw_folder_path_parent = os.path.join(real_path, "RAW_bases")
config_path["raw_dir"] = ["ALASKA2_Base", "Boss_Base", "Dresden_Base", "RAISE_Base", "StegoApp_Base",
                          "Wesaturate_Base"]
# config_path["raw_dir"] = ["ALASKA2_Base", "Boss_Base"]
# where JPEG images will be stored:
config_path["out_dir"] = config_path["root"] + "/" + baseName + "_JPG_1024x1024_QF75"
# where JPEG images split with multi crop will be stored:
config_path["out_dir_multisplit"] = config_path["root"] + "/" + baseName + "_JPG_MultiSplit_256x256_QF75"
# where TIFF (uncompressed) images will be stored:
config_path["out_dir_tif"] = config_path["root"] + "/" + baseName + "_TIFF_1024x1024"

# where intermediate (temporary) images will be stored:
config_path["tmp_dir"] = config_path["root"] + "/TIFF_tmp"
# important, direcroty in which "profiles" that defines the development parameters will be written for each and every
# image:
config_path["profile_used_dir"] = config_path["root"] + "/profiles_applied"
# initial profiles, for demosaicing only:
config_path["dem_profile_dir"] = "demProfiles"

# File in which the randomly generated development parameters are output for loging purpose:
backup_file_path = config_path["root"] + "/list_img_profiles.txt"

# Second main variable, the "config_process", that defines, for ALL development parameters, the range in which
# those are picked. This configuration of the development process is quite "coarse grain"; More specification on
# the distribution of each parameters are to be found in the companion script "random_dev.py"
config_process = dict()
# Number of JPG image to create; if larger than number of RAW images automatically reset to the latest ;
# otherwise, randomly select a subset of images
config_process["number_of_output_images"] = 100000000
config_process["jpg_per_raw"] = 16
# Probability of using unsharpening mask
config_process["prob_usm"] = 1
# Probability of using directional pyramid denoising algorithm
config_process["prob_denoise"] = 1
# Probability of using unsharpening mask if denoising is used first
config_process["prob_usm_if_denoise"] = 0
# Probability of using denoising if unsharpening mask is used first
config_process["prob_denoise_if_usm"] = 1

# Possible final image size (a very last cropping step is applied)
config_process["crop_size"] = [512, 640, 720, 1024]
# config_process["crop_size"] = 1024
# If this variable is None: resize with a random factor, else: resize with the size write below
config_process["resize_size"] = 1024
# To match this final size, we can either crop / resize or do both; those are used with the following probabilities:
# Probability of resizing images:
config_process["prob_resize_only"] = 0
# Probability of croping the image:
config_process["prob_crop_only"] = 0
# Probability of doing both resising and then croping:
config_process["prob_resize_and_crop"] = 1 - config_process["prob_resize_only"] - config_process["prob_crop_only"]
# Definition of the set of possible resampling kernels (for resizing)....
config_process["resize_kernel"] = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
# config_process["resize_kernel"] = Image.LANCZOS
# Along with the probability of each
config_process["resize_kernel_prob"] = [0.1, 0.15, 0.25, 0.5]
# config_process["resize_kernel_prob"] = 1
# Maximal resizing factor (here, upsampling by 30%)
config_process["resize_factor_upperBound"] = 1.30
# QF
# config_process["jpeg_qf"] = np.arange(60, 100 + 1)
config_process["jpeg_qf"] = 75
# Probabilities corresponding of QF
config_process["jpeg_qf_probabilities"] = [0.0030, 0, 0, 0, 0, 0.0010, 0, 0, 0, 0.0010, 0.0070, 0.0020, 0.0010, 0,
                                           0.0010, 0.1080, 0.0010, 0.0010, 0.0010, 0.0020, 0.1730, 0.0020, 0.0020,
                                           0.0020, 0.0040, 0.0840, 0.0050, 0.0040, 0.0100, 0.0180, 0.1190, 0.0090,
                                           0.0160, 0.0220, 0.0240, 0.0510, 0.0470, 0.0350, 0.0560, 0.0300, 0.1580]
# config_process["jpeg_qf_probabilities"] = 1
# Things are slightly different for demosaicing since each demosaicing algorithm needs to be associated with a pp3 file.
# Hence, we first set the directory that contain the files for different demosaicing algorithms ....
config_process["demosaicing"] = os.listdir(config_path["dem_profile_dir"])
# then we need to set the probability of each of those (that, of course, needs to be a vector with number of component
# equals the number of demosaicing files);
dem_probs = []
for dem_name in config_process["demosaicing"]:
    if dem_name == "dem_igv.pp3":
        dem_probs.append(0.1)
    if dem_name == "dem_amaze.pp3":
        dem_probs.append(0.4)
    if dem_name == "dem_fast.pp3":
        dem_probs.append(0.15)
    if dem_name == "dem_dcb_2_amel.pp3":
        dem_probs.append(0.35)
config_process["demosaicing_probabilities"] = dem_probs


# **************************#
# MAIN conversion function #
# **************************#
def From_RAW_to_JPG(RAWimageName, RAWpath):
    # Here we start 1) splitting image path by filename and extension
    imageBaseName = os.path.splitext(RAWimageName)[0]
    imageRawExtension = os.path.splitext(RAWimageName)[1]

    # First check to modify the TIFimagePath in order to develop more than only one image per RAW
    if os.path.exists(config_path["profile_used_dir"] + "/" + imageBaseName + ".pp3"):
        counter = 0
        # print("Reference: " + imageBaseName)
        for complete_name in os.listdir(config_path["out_dir"]):
            name = os.path.splitext(complete_name)[0].split('_')[0]
            # print(name)
            if name == imageBaseName:
                counter += 1

        # print("Finale counter = " + str(counter))
        imageBaseName = imageBaseName + "_" + str(counter + 1)

    # and 2) create path for all temporary image.
    TIFimagePath = os.path.join(config_path["tmp_dir"], imageBaseName + "_tmp.tif")
    TIFimage2Path = os.path.join(config_path["tmp_dir"], imageBaseName + "_tmp2.tif")
    TIFimage3Path = os.path.join(config_path["out_dir_tif"], imageBaseName + ".tif")
    RAWimagePath = os.path.join(RAWpath, imageBaseName.split('_')[0] + imageRawExtension)
    ImageProfilePath = os.path.join(config_path["profile_used_dir"], imageBaseName + ".pp3")
    print("Converting Image " + RAWimagePath)

    if bool_random_dev:
        # We create, for each and every images, a random generator that will be used to create (randomly) a development
        # process file. To ensure the randomness and reproducibility of the development process, we propose to seed
        # each generator by the MD5 hashsum of image filename To generate several version of the same dataset you can
        # use, for instance, the following commands (which hash a value from system time to generate a random seed for
        # every image)
        # imageSeed=int.from_bytes(md5( (round(time.time() * 100000)**2).to_bytes(32, byteorder='big') ).digest(),
        # 'big') % 2**32
        imageSeed = int.from_bytes(md5(bytes(imageBaseName, 'utf-8')).digest(), 'big') % 2 ** 32
        # imageSeed = None
        rg = devRandomGenerator(config_process["jpeg_qf"], config_process["jpeg_qf_probabilities"],
                                config_process["crop_size"], config_process["demosaicing"],
                                config_process["demosaicing_probabilities"], config_process["resize_kernel"],
                                config_process["resize_kernel_prob"], seed=imageSeed)
    else:
        rg = devFixGenerator(
            config_process["jpeg_qf"],
            config_process["resize_size"],
            config_process["demosaicing"]
        )

    # The very first step consists in generating a random development file for the given image ; thus, if such a file
    # exists, the associated image has already been processed. This is used to allows a cheap, yet efficient
    # parallelization by simply launching several time the same conversion script (see Section "Parallelization" in
    # the pdf documentation)
    if not os.path.exists(ImageProfilePath):
        # INITIALIZATION: random selection of development / processing parameters and storing into a pp3 file
        DevList = {
            "name": imageBaseName,
            "dem": rg.dem["dem_algorithm"](),
            "subsampling_type": rg.r.choice([0, 1, 2], p=[config_process["prob_resize_and_crop"],
                                                          config_process["prob_resize_only"],
                                                          config_process["prob_crop_only"]]),
            "resize_kernel": rg.resize_kernel["kernel"](),
            "resize_weight": rg.resize_weight["factor"](),
            "crop_size": rg.crop["size"](),
            "qf": rg.QF["QF"]()
        }
        if bool_random_dev:
            DevList["choice"] = {
                "usm": rg.r.binomial(1, config_process["prob_usm"]),
                "denois": rg.r.binomial(1, config_process["prob_denoise"]),
                "usm_if_denois": rg.r.binomial(1, config_process["prob_usm_if_denoise"]),
                "denois_if_usm": rg.r.binomial(1, config_process["prob_denoise_if_usm"])
            }
        else:
            DevList["choice"] = {
                "usm": config_process["prob_usm"],
                "denois": config_process["prob_denoise"],
                "usm_if_denois": config_process["prob_usm_if_denoise"],
                "denois_if_usm": config_process["prob_denoise_if_usm"]
            }

        dumpFile = open("/tmp/dumpOutPut.txt", "wb")
        # this file is used to dump output from rawtherapee and x3f_extract which are quite verbose and cannot be
        # used in quiet mode :(
        # FIRST STEP: APPLYING DEMOSAICING ! Note that, we used rawtherapee version 5.7 which seems, as opposed to
        # version 5.3, to handle efficiently X3F Sigma foveon trichromatic sensor
        if imageRawExtension.upper() == ".X3F":
            # However, some X3F images still cannot be processed with rawtherapee; for this reason we try first to
            # apply rawtherappe; if it fails, we call x3f_extractor executable.
            print("[WARNING] Sigma Foveon X3F raw file ! Trying RawTherapee")
            # This is a typical use of call to execute the rawtherapee-cli command (note that the output are dumped
            # to /tmp/ )
            call(["rawtherapee-cli", "-a", "-q", "-t", "-b16", "-o", TIFimagePath, "-p",
                  os.path.join(config_path["dem_profile_dir"], DevList["dem"]), "-c", RAWimagePath], stdout=dumpFile,
                 stderr=dumpFile)
            # This is the "if rawtherapee fails" which is tested as "if not image file is generated"
            if not os.path.exists(TIFimagePath):
                # This is a typical use of binary x3f_extract to dump tiff data from X3F file (note that the output
                # are dumped to /tmp/ )
                call(["./x3f_extract", "-q", "-tiff", "-no-denoise", "-no-sgain", RAWimagePath], stdout=dumpFile,
                     stderr=dumpFile)
                # This script automatically write output image into the same directory, we move this file to match
                # TIFimagePath variable
                shutil.move(os.path.join(RAWimagePath + ".tif"), TIFimagePath)
            if not os.path.exists(TIFimagePath):
                print("[ERROR] neither rawtherapee nor x3f_extract managed to read this file! Are you sure it is not "
                      "corrupted ?!?")
                # if not X3F raw image files, we call also rawtherapee

        # if not X3F raw image files, we call also rawtherapee
        else:
            call(["rawtherapee-cli", "-a", "-q", "-t", "-b16", "-o", TIFimagePath, "-p",
                  os.path.join(config_path["dem_profile_dir"], DevList["dem"]), "-c", RAWimagePath],
                 stdout=dumpFile, stderr=dumpFile)
        # Before moving forward, we ensure that the TIF image (resulting for demosaicing of RAW) does exist; indeed some
        # raw images files format cannot be read.
        if os.path.exists(TIFimagePath):

            # SECOND STEP: RESIZING and CROPPING
            # First of all, we carry out the resizing ; thi requires one extra parameter (the resizing factor) that
            # depends on the image size ;
            # To deal with this we call the resizing and get the factor as an output ....
            DevList["subsampling_factor"] = imProc.image_randomize_resizing(TIFimagePath, TIFimage2Path,
                                                                            DevList['crop_size'][0],
                                                                            DevList['crop_size'][1],
                                                                            subsampling_type=DevList[
                                                                                'subsampling_type'],
                                                                            kernel=DevList['resize_kernel'],
                                                                            resize_weight=DevList['resize_weight'],
                                                                            resize_factor_UB=config_process[
                                                                                "resize_factor_upperBound"],
                                                                            resize_size=config_process["resize_size"],
                                                                            grayscale=bool_grayscale)

            # ... Then, and only then, we can write dump the profiles of the image in the associated file.
            if bool_random_dev:
                rg.generate_random_RT_profile(
                    imageDevList=DevList,
                    outputPath=ImageProfilePath,
                    backupfile=backup_file_path
                )
            else:
                rg.generate_fix_RT_profile(
                    imageDevList=DevList,
                    outputPath=ImageProfilePath,
                    backupfile=backup_file_path,
                    prob_usm_if_denoise=config_process["prob_usm_if_denoise"]
                )

            if os.path.exists(TIFimage2Path):
                # FORTH (and main) STEP: generating processing pipeline file and using rawtherapee
                call(["rawtherapee-cli", "-a", "-q", "-t", "-b8", "-o", TIFimage3Path, "-p", ImageProfilePath, "-c",
                      TIFimage2Path], stdout=dumpFile, stderr=dumpFile)

                if os.path.exists(TIFimage3Path):  # All JPEG in same folder but different database
                    raw_folder = os.path.split(RAWpath)[1]
                    if bool_multicrop and bool_random_dev is False:
                        # Split the TIFF in x TIFF of 256x256
                        multi_crop(TIFimage3Path, config_process["jpg_per_raw"])

                        tif_multicrop_path = os.path.join(config_path["out_dir_tif"], "Multi_Crop", imageBaseName)
                        # Save JPEG in different folder (each RAW folder have 16 JPEG images)
                        # jpeg_mutlicrop_path = os.path.join(config_path["out_dir"], "Multi_Crop", imageBaseName)
                        jpeg_mutlicrop_path = os.path.join(config_path["out_dir_multisplit"], raw_folder)
                        if not os.path.exists(jpeg_mutlicrop_path):
                            os.makedirs(jpeg_mutlicrop_path, 0o755)
                        for i in range(config_process["jpg_per_raw"]):
                            imProc.jpeg_compression(
                                infile=os.path.join(tif_multicrop_path, imageBaseName + "_" + str(i + 1) + ".tif"),
                                outpath=os.path.join(jpeg_mutlicrop_path, imageBaseName + "_" + str(i + 1) + ".jpg"),
                                qf=DevList["qf"])

                    # LAST STEP: (mere) jpeg compression
                    jpeg_path = os.path.join(config_path["out_dir"], raw_folder)
                    if not os.path.exists(jpeg_path):
                        os.makedirs(jpeg_path, 0o755)
                    imProc.jpeg_compression(infile=TIFimage3Path,
                                            outpath=os.path.join(jpeg_path, imageBaseName + ".jpg"),
                                            qf=DevList["qf"])

                    # Eventually, we double check that the associated JPEG image exists;
                    # if not we keep the TIF temporary files for backup and debugging
                    if os.path.exists(os.path.join(jpeg_path, imageBaseName + ".jpg")):
                        # We can either keep tiff (uncompressed) image
                        if keepUncompressed:
                            call(["rm", TIFimagePath, TIFimage2Path])
                            print("[SUCCESS] Images ", imageBaseName + ".tif and", imageBaseName + ".jpg",
                                  " Converted successfully ")
                        # or keep only the jpg, in such case, we remove ALL itermediate images
                        else:
                            tif_multicrop_folder_path = os.path.join(
                                config_path["out_dir_tif"],
                                "Multi_Crop",
                                imageBaseName)
                            if os.path.exists(tif_multicrop_folder_path) \
                                    and len(os.listdir(tif_multicrop_folder_path)) == config_process["jpg_per_raw"]:
                                shutil.rmtree(tif_multicrop_folder_path)

                            # When the JPEG is saved, delete all TIF corresponding to this image
                            call(["rm", TIFimagePath, TIFimage2Path, TIFimage3Path])
                            print("[SUCCESS] Image ", imageBaseName + ".jpg", " Converted successdully ")

                    # Print out possible causes that lead not to develop the given RAW images, for logging.
                    else:
                        print("[ERROR] Ultimate JPEG FAILED FOR" + RAWimagePath)
                else:
                    print("[ERROR] Last conversion (RAWTHERAPEE) FAILED FOR" + RAWimagePath)
            else:
                print("[ERROR] SUBSAMPLING FAILED FOR" + RAWimagePath)
        else:
            print("[ERROR] Image " + RAWimagePath + " can hardly be converted to TIFF: skipped")
    else:
        print("[WARNING] Image: " + imageBaseName + ".jpg already processed: skipped ")


def multi_crop(initial_path, nb_images):
    path = os.path.splitext(initial_path)[0]
    raw_image = str.split(path, '/')[2]
    path = os.path.join(str.split(path, '/')[0], str.split(path, '/')[1])
    complete_path = os.path.join(path, "Multi_Crop", raw_image)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path, 0o755)

    # im = (tifffile.imread(initial_path) / (2 ** 16 - 1))
    im = tifffile.imread(initial_path)
    img_width, img_height = im.shape[0:2]

    step = np.sqrt(nb_images)
    step_height = int(np.ceil(img_height / step))
    step_width = int(np.ceil(img_width / step))

    k = 0
    imgs = []
    for i in range(0, img_height, step_height):
        for j in range(0, img_width, step_width):
            # box = (j, i, min(img_width, j + step_width), min(img_height, i + step_height))
            # a = im.crop(box)

            # Here we simply compute the first and last indices of pixels' central area.
            left = i
            top = j
            right = min(img_width, i + step_width)
            bottom = min(img_height, j + step_height)

            imagette = im[left:right, top:bottom, :]
            try:
                tifffile.imwrite(complete_path + "/" + raw_image + "_" + str(k + 1) + ".tif", imagette)
                imgs.append(imagette)
            except:
                print("Error somewhere when the image is being save...")
                pass
            k += 1

            # return imgs


# **************************#
#  BEGINNING OF THE SCRIPT  #
# **************************#

if __name__ == '__main__':
    # The beginning of the script, we get the time
    start_time = time.time()

    # First of all, we check out if some specified directories need to be created and do so.
    for d in config_path:
        # Remove part 264-268
        if bool_remove_beginning and d in remove_dir and os.path.exists(config_path[d]):
            shutil.rmtree(config_path[d])

        if d is not "raw_dir" and not os.path.exists(config_path[d]):
            # Check if it's necessary to create a directory "multicrop"
            if d is "out_dir_multisplit" and bool_multicrop and bool_random_dev:
                print("Multi crop with a random development is not allowed,"
                      "so the directory for multi crop images isn't create")
            else:
                # Creates the directory with classic permissions
                print("Created --> " + config_path[d])
                os.makedirs(config_path[d], 0o755)

            # List of RAW images into the specified directory
            # RAWimagesName = os.listdir(config_path["raw_dir"])

    if os.path.exists(backup_file_path):
        os.remove(backup_file_path)

    # For each folder in raw_dir
    for raw_path in config_path["raw_dir"]:
        RAWimagesName = sorted(os.listdir(os.path.join(raw_folder_path_parent, raw_path)))

        # Random selection of a subset of images (the total number of image picked is specified in config_process
        # --> number_of_output_images)
        # We selected random indices
        image_indices = np.arange(len(RAWimagesName))
        np.random.shuffle(image_indices)
        image_indices = image_indices[0:min(config_process["number_of_output_images"], len(RAWimagesName) * 16)]
        print("Number of images to be created/converted : ",
              min(config_process["number_of_output_images"], len(RAWimagesName) * config_process["jpg_per_raw"]))

        # The script can be launched using multiprocessing
        # Default configuration is to use half of the number of cores ... you can set this value to something higher
        # (Remi used 3/4 of total number of cores)
        # numCores = int(multiprocessing.cpu_count() / 2)
        numCores = int(multiprocessing.cpu_count() * 2 / 3)
        # numCores = int(multiprocessing.cpu_count() * 3 / 4)
        Parallel(n_jobs=numCores, verbose=1)(
            delayed(From_RAW_to_JPG)(
                RAWpath=os.path.join(raw_folder_path_parent, raw_path),
                RAWimageName=RAWimagesName[index]) for index in image_indices)

    # At the end of the script we get the time too and make the difference between the start_time and now
    print("\nTime to create the all base: " + str(datetime.timedelta(seconds=round(time.time() - start_time))))

# This alternative consists is the same processing ... only without multiprocessing
# Instead, we simply loop over all images
# for index in imageIndices:
# From_RAW_to_JPG(RAWimageName=RAWimagesName[index])
