# adapted from code by Sunay Gotla
import sys
import SimpleITK as sitk
import os
import numpy as np
import glob
'''

to run at the command line

python skull_strip.py ncct_location out_location

ncct_location = the location of the NCCT dicoms
out_location = location where you want to save processed images

outfiles are as follows:

ncct.nii.gz: NIFTI file for the NCCT
kMeans.nii.gz: output of the KMeans segmentation (downsampled)
levelSets.nii.gz: output of Level Sets segmentation (downsampled)
levelSets_morph.nii.gz: segmentation with morphological operations (downsampled)
brain_mask.nii.gz: final brain mask
ncct_stripped.nii.gz: final result, original image with skull stripped

'''

def main(ncct_location, out_location):
    if not os.path.exists(out_location):
        os.makedirs(out_location)

    dcm_names = glob.glob(os.path.join(ncct_location, '*'))

    dcm_names = [file for file in dcm_names
                 if '.dbi' not in file]

    reader = sitk.ImageFileReader()
    f_names_pos = []

    for f in dcm_names:
        reader.SetFileName(f)
        reader.ReadImageInformation()
        f_names_pos.append((f, float(reader.GetMetaData('0020|0032').split('\\')[2])))
        f_names_pos.sort(key=lambda x: x[1])
        f_names_sorted, _ = zip(*f_names_pos)

    fixed = sitk.ReadImage(f_names_sorted)
    sitk.WriteImage(fixed, os.path.join(out_location, 'ncct.nii.gz'))

    ################
    # DOWNSAMPLE
    ################

    fixedSmall = sitk.Shrink(fixed, [3, 3, 3])

    minPixelSpacing = min(fixedSmall.GetSpacing())
    dimension = fixedSmall.GetDimension()
    timeStep = minPixelSpacing / (2 ** (dimension + 1))

    fixedSmooth = sitk.CurvatureAnisotropicDiffusion(
        sitk.Cast(fixedSmall, 8),
        timeStep=timeStep,
        conductanceParameter=5,
        numberOfIterations=5)

    ################
    # K-MEANS BASED INITIAL BRAIN SEGMENTATION
    ################

    outputImage = fixedSmooth*0
    i = 1
    counts = [0, 0]
    while ((counts[1] > 0.5 * counts[0]
            or counts[1] == 0
            or counts[0] < 1000)
           and i <= fixedSmooth.GetSize()[2]):
        im_slice = fixedSmooth[:, :, -i] * sitk.Cast(fixedSmooth[:, :, -i] > 0, 8)
        test = sitk.ScalarImageKmeans(im_slice, list(range(0, 5)), False)
        test_slice = sitk.JoinSeries(test)
        #     test_slice = sitk.BinaryFillhole(test_slice == 1)
        outputImage = sitk.Paste(
            sitk.Cast(outputImage, 1),
            test_slice,
            test_slice.GetSize(),
            destinationIndex=[0, 0, outputImage.GetSize()[2] - (i - 1)])
        # COUNTING THE NUMBER OF NON-ZERO PIXELS IN EACH SLICE
        # TO LOCATE THE FATTEST SLICE
        if np.count_nonzero(sitk.GetArrayFromImage(test == 1)) > counts[0]:
            counts[0] = np.count_nonzero(sitk.GetArrayFromImage(test == 1))
            maxSlice = i
        #     counts[2] = counts[1] #Previous count
        counts[1] = np.count_nonzero(sitk.GetArrayFromImage(test == 1))
        #     if counts[2] == 0:
        #         counts[2] = np.count_nonzero(sitk.GetArrayFromImage(test==1))
        i += 1
        print(counts[0], counts[1])

    # calculate z location of stopping point
    z_location = fixedSmooth.TransformIndexToPhysicalPoint([0, 0, fixedSmooth.GetSize()[2] - maxSlice])[2]

    print('Z-location of stopping point: {}'.format(z_location))
    outputImage = sitk.BinaryErode(outputImage, (2, 2, 2), sitk.sitkBall)
    sitk.WriteImage(outputImage == 1, os.path.join(out_location, 'kMeans.nii.gz'))

    ################
    # LEVEL SETS BASED EXPANSION OF INITIAL BRAIN SEGMENTATION
    ################

    def progressBar(percent):
        import sys
        bar_len = 60
        filled = int(round(bar_len * percent))
        bar = '|' * filled + '-' * (bar_len - filled)  # print full block character
        sys.stdout.write('[%s]: %.1f%s \r' % (bar, round(percent * 100, 3), '%'))
        sys.stdout.flush()

    MAX_THREADS = 4
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

    init_ls = sitk.SignedMaurerDistanceMap(outputImage == 1, True, True, True)
    # sitk.WriteImage(sitk.Cast(init_ls, sitk.sitkInt16), 'init_ls_CSF.nii.gz')
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(5)
    lsFilter.SetUpperThreshold(300)
    lsFilter.SetMaximumRMSError(1e-6)
    lsFilter.SetNumberOfIterations(1000)
    lsFilter.SetCurvatureScaling(1)
    lsFilter.SetPropagationScaling(1)
    lsFilter.ReverseExpansionDirectionOn()
    lsFilter.AddCommand(sitk.sitkProgressEvent, lambda: progressBar(lsFilter.GetProgress()))

    ls = lsFilter.Execute(init_ls, sitk.Cast(fixedSmall, init_ls.GetPixelID()))

    lsBin = sitk.BinaryThreshold(ls, lowerThreshold=0, upperThreshold=100)
    lsBin = sitk.OpeningByReconstruction(lsBin, (6, 6, 6), sitk.sitkBall)
    fixedSmall_masked = sitk.Cast(lsBin, 8) * fixedSmooth
    fixedSmall_masked = sitk.BinaryThreshold(fixedSmall_masked, lowerThreshold=0.1, upperThreshold=250)
    lsBin = fixedSmall_masked > 0
    sitk.WriteImage(lsBin, os.path.join(out_location, 'levelSets.nii.gz'))
    # morphological operations
    lsBin = sitk.BinaryFillhole(lsBin)
    lsBin = sitk.BinaryMorphologicalOpening(lsBin, (2, 2, 2), sitk.sitkBall)
    lsBin_filled = sitk.BinaryMorphologicalClosing(lsBin, (2, 2, 2), sitk.sitkBall)
    lsBin_filled = sitk.BinaryErode(sitk.BinaryFillhole(lsBin_filled), (1, 1, 1), sitk.sitkBall)
    sitk.WriteImage(lsBin_filled, os.path.join(out_location, 'levelSets_morph.nii.gz'))

    # Set the last slice to 0 (top of skull in case of poor patient placement)
    zeroedSlice = sitk.JoinSeries(lsBin_filled[:, :, 0] * 0)
    lsBin_filled = sitk.Paste(
        lsBin_filled,
        zeroedSlice,
        zeroedSlice.GetSize(),
        destinationIndex=[0, 0, lsBin_filled.GetSize()[2] - 1])

    ################
    # UPSAMPLE
    ################

    ref = fixed
    ls_bin_out = sitk.Resample(lsBin_filled, ref.GetSize(),
                               sitk.Transform(),
                               sitk.sitkNearestNeighbor,
                               ref.GetOrigin(),
                               ref.GetSpacing(),
                               ref.GetDirection(),
                               0,
                               ref.GetPixelID()) # lsBig = sitk.Expand(lsBin, oldSize)
    sitk.WriteImage(ls_bin_out, os.path.join(out_location, 'brain_mask.nii.gz'))

    ################
    # EXTRACT BRAIN
    ################
    sitk.WriteImage(fixed*sitk.Cast(ls_bin_out, 2), os.path.join(out_location, 'ncct_stripped.nii.gz'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
