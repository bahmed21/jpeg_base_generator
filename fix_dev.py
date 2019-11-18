import numpy as np
import shutil
from PIL import Image

# Script used to randomly select the development parameters from RAW files to JPEG images.
# This merely consists of a initializer ( the function __init__ ) and a "development profile" random generator.
# The initializer is used to set all the development parameters; for simplicity we recall that those are the following:
#   1) "dem_algorithm"  --> the demosaicing algorithm
#   2) "resize"         --> a (sometimes used) resizing factor
#   3) "denoise"         --> directional pyramid denoising settings
#                       --> We use two parameters here "luminance" which corresponds to the denoising strength and
#                           "detail" which is defines, roughly speaking, the level of edge preserving
#   4) sharpening (with Unsharpening mask) settings
#                       --> We use two parameters there "radius" which corresponds to the size of the (convolution)
#                           unsharpening mask and "amount" which correspond to the amount of unsharpening the denoising
#                           strength and "detail" which is defines, roughly speaking, the level of edge preserving
#   5) Edge enhancement tools (micro-contrast) settings
#                       --> We use two parameters there "radius" which corresponds to the size of the (convolution)
#                           unsharpening mask and "amount" which correspond to the amount of unsharpening the denoising
#                           strength and "detail" which is defines, roughly
# More details can be found at: https://rawpedia.rawtherapee.com
#
# Eventually, outside from rawtherapee software, we also specifies:
#   6) JPEG compression quality factor
#   7) final image size with of course two parameters.

KERNEL_dict = {Image.NEAREST: "NEAREST", Image.BILINEAR: "BILINEAR", Image.BICUBIC: "BICUBIC", Image.LANCZOS: "LANCZOS"}


class devFixGenerator:
    # Initializer whose main goal is to define the statistical distributions for all the parameters considered
    # **************************#
    # Main function: initializer #
    # **************************#
    # The goal of this function is to select randomly  function
    def __init__(self, qf, crop_size, dem, seed=None):
        self.r = np.random.RandomState(seed)

        self.dem = {"dem_algorithm": lambda: "dem_amaze.pp3"}

        self.usm = {"radius": lambda: 0.5,
                    "amount": lambda: 250}

        self.denois = {"luminance": lambda: 0,
                       "detail": lambda: 50}

        # self.microcontrast = {"quantity": lambda: min(round(self.r.gamma(1, 0.5) * 100), 100),
        #                       "uniformity": lambda: max(0, round(self.r.normal(30, 5)))}

        self.QF = {"QF": lambda: qf}

        # self.crop = {"size": lambda: self.r.choice(crop_size, 2)}
        self.crop = {"size": lambda: [crop_size, crop_size]}
        self.resize_kernel = {"kernel": lambda: Image.LANCZOS}
        self.resize_weight = {"factor": lambda: 0}

    # Random profile according to the probabilities associated with each development step, as step in the variable
    # process_config from the main script ALASKA_conversion.py, we pick, or not, a random value for each parameter
    # following the distribution defined in the initializer The development process is eventually written into a
    # rawtherapee compatible pp3 file.
    def generate_fix_RT_profile(self, imageDevList, outputPath, backupfile, prob_usm_if_denoise):
        radius = 0
        amount = 0
        luminance = 0
        detail = 0
        USM_before_DENOISE = 1

        # This is the pp3 file in which development parameters will be written for later used in rawtherapee.
        currentProfile = open(outputPath, 'w+')
        # Writing of the header of pp3 file.
        currentProfile.write("[Version]\nAppVersion=5.4\nVersion=331\n\n")
        # Specifies if denoising is applied prior or after sharpening.
        if prob_usm_if_denoise < 0.5:
            # There we start we unsharpening mask and pick randomly the associated parameters (radius and amount)
            if imageDevList["choice"]["usm"] == 1:
                radius = self.usm["radius"]()
                amount = self.usm["amount"]()
                currentProfile.write(
                    "[Sharpening]\nEnabled=true\nMethod=usm\nRadius={}"
                    "\nAmount={}\nThreshold=20;80;2000;1200;\n\n".format(radius, amount))

                # and, in needed, specifies the parameters for the denoising
                if imageDevList["choice"]["denois_if_usm"] == 1:
                    luminance = self.denois["luminance"]()
                    detail = self.denois["detail"]()
                    currentProfile.write("[Directional Pyramid Denoising]\nEnabled=true\nEnhance=false\nMedian=false"
                                         "\nLuma={}\nLdetail={}\n\n".format(luminance, detail))

                # there the steps are applied in the other way round, i.e denoising first ....
        else:
            USM_before_DENOISE = 0
            if imageDevList["choice"]["denois"] == 1:
                luminance = self.denois["luminance"]()
                detail = self.denois["detail"]()
                currentProfile.write("[Directional Pyramid Denoising]\nEnabled=true\nEnhance=false\nMedian=false"
                                     "\nLuma={}\nLdetail={}\n\n".format(luminance, detail))
                # ... and then unsharpening mask.
                if imageDevList["choice"]["usm_if_denois"] == 1:
                    radius = self.usm["radius"]()
                    amount = self.usm["amount"]()
                    currentProfile.write("[Sharpening]\nEnabled=true\nMethod=usm\nRadius={}\nAmount={}"
                                         "\nThreshold=20;80;2000;1200;\n".format(radius, amount))

        currentProfile.close()

        # Optional: one can log all development parameters for all images into a single file. If so we write into a
        # slightly more verbose the development parameters
        if backupfile is not None:
            BackupProfile = open(backupfile, 'a+')
            if radius == 0 and amount == 0:
                USM_set = "OFF"
            else:
                USM_set = "ON"

            if luminance == 0 and detail == 0:
                DENOISE_set = "OFF"
            else:
                DENOISE_set = "ON"

            RESIZE_factor = imageDevList["subsampling_factor"]
            if imageDevList["subsampling_type"] == 0:
                RESIZE_set = "ON_WITH_CROP"
                RESIZE_kernel = KERNEL_dict[imageDevList["resize_kernel"]]
            elif imageDevList["subsampling_type"] == 1:
                RESIZE_set = "ON_ALONE"
                RESIZE_kernel = KERNEL_dict[imageDevList["resize_kernel"]]
            elif imageDevList["subsampling_type"] == 2:
                RESIZE_set = "CROP_ONLY"
                RESIZE_kernel = "NONE"
            else:
                RESIZE_set = "OFF"

            # The line written into the log file
            BackupProfile.write(
                "%30s | DEM %20s | %d | USM %3s , %5.2f , %6.2f | DENOISE %3s , %5.2f , %5.2f | "
                "RESIZE %s , %s , %5.5f | %4d x %4d | %3d \n" % (
                    imageDevList["name"],
                    imageDevList["dem"],
                    USM_before_DENOISE,
                    USM_set, radius, amount,
                    DENOISE_set, luminance, detail,
                    RESIZE_set, RESIZE_kernel, RESIZE_factor,
                    imageDevList["crop_size"][0], imageDevList["crop_size"][1],
                    imageDevList["qf"])
            )

        BackupProfile.close()
