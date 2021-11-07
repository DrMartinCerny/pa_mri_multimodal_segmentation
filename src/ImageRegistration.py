import SimpleITK as sitk

class ImageRegistration:

    def __init__(self, config):
        self.config = config

    def findTransformation(self, fixed, moving):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, self.config.IMAGE_REGISTRATION_EPOCHS )
        R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)
        return R.Execute(fixed, moving)