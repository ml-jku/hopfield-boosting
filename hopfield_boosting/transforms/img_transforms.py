import random
import numpy as np
from PIL import Image


class BWToRandColor(object):
    """
    Transform to replace black and white pixels with random colors in an RGB image.

    Methods:
        __call__(self, img: Image.Image) -> Image.Image: Apply the transformation to the input image.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with replaced black and white pixels.
        """
        if img.mode == "RGB":
            # Generate random colors for black and white replacement
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Convert image to NumPy array
            img_array = np.array(img)

            # Create a mask for black pixels
            black_mask = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)

            # Create a mask for white pixels
            white_mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)

            # Replace black pixels with random color1
            img.paste(color1, None, mask=Image.fromarray(black_mask))

            # Replace white pixels with random color2
            img.paste(color2, None, mask=Image.fromarray(white_mask))

            # Convert the NumPy array back to a PIL image
            img = Image.fromarray(np.uint8(img))

        return img


class GrayToRandColor(object):
    """
    Transform to interpolate colors based on grayscale intensity in an RGB image.

    Methods:
        __call__(self, img: Image.Image) -> Image.Image: Apply the transformation to the input image.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with colors interpolated based on grayscale intensity.
        """
        if img.mode == "RGB":
            # Generate random colors for smooth fade
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Convert image to NumPy array
            img_array = np.array(img)

            # Calculate grayscale intensity
            gray_intensity = np.mean(img_array, axis=-1)

            # Normalize the intensity to range [0, 1]
            normalized_intensity = gray_intensity / 255.0

            # Interpolate between color1 and color2 based on intensity
            interpolated_colors = (
                (1 - normalized_intensity) * color1[0] + normalized_intensity * color2[0],
                (1 - normalized_intensity) * color1[1] + normalized_intensity * color2[1],
                (1 - normalized_intensity) * color1[2] + normalized_intensity * color2[2]
            )

            # Replace pixels with interpolated colors
            img_array[:, :, 0] = interpolated_colors[0]
            img_array[:, :, 1] = interpolated_colors[1]
            img_array[:, :, 2] = interpolated_colors[2]

            # Convert the NumPy array back to a PIL image
            img = Image.fromarray(np.uint8(img_array))

        return img        