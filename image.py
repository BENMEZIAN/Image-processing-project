import tkinter as tk
from tkinter import filedialog
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


class ImageHistogram:
    def __init__(self, master):
        self.master = master
        master.title("Image operations")
        master.geometry("900x550")
        
        self.img = None
        
        # create a button to upload image
        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        # create a button to display histograms
        self.show_histogram_button = tk.Button(master, text="Show Histogram", command=self.show_histogram, state="disabled")
        self.show_histogram_button.pack(pady=10)
        
        # create a button to perform histogram equalization
        self.histeq_button = tk.Button(master, text="Perform Histogram Equalization", command=self.histogram_equalization)
        self.histeq_button.pack(pady=10)
        
        # create a button to perform expansion
        self.expansion_button = tk.Button(master, text="Perform Expansion image", command=self.expansion_image)
        self.expansion_button.pack(pady=10)
        
        # create a button to perform translation
        self.translation_button = tk.Button(master, text="Perform translation image", command=self.translation_image)
        self.translation_button.pack(pady=10)
        
        # create a button to perform inversion
        self.inversion_button = tk.Button(master, text="Perform inversion image", command=self.histogram_inversion)
        self.inversion_button.pack(pady=10)
        
        # create a button to perform histogram color
        self.histogramcolors_button = tk.Button(master, text="Histogram colors", command=self.hitogram_colors)
        self.histogramcolors_button.pack(pady=10)
        
        # create a button to perform median filter
        self.median_filter_button = tk.Button(master, text="Median filter", command=self.median_filter)
        self.median_filter_button.pack(pady=10)
        
        # create a button to perform gaussian filter
        self.gaussian_filter_button = tk.Button(master, text="Gaussian filter", command=self.gaussian_filter)
        self.gaussian_filter_button.pack(pady=10)
        
        # create a button to perform sobel filter
        self.sobel_filter_button = tk.Button(master, text="Sobel filter", command=self.sobel_filter)
        self.sobel_filter_button.pack(pady=10)
        
        # create a button to perform binarization
        self.binarization_button = tk.Button(master, text="Binarization", command=self.binarization)
        self.binarization_button.pack(pady=10)
        
        # create a button to perform quantization
        self.quantization_button = tk.Button(master, text="Quantification", command=self.quantize_colors)
        self.quantization_button.pack(pady=10)
        
        
    def upload_image(self):
        self.filename = filedialog.askopenfilename(initialdir=".", title="Select Image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
        if self.filename:
            self.img = Image.open(self.filename) # Convert to grayscale
            self.show_histogram_button.config(state="normal")
            
    def show_histogram(self):
        if self.img:
            # Calculate histogram, cumulative histogram, normalized histogram, and normalized cumulative histogram
            histogram = [0]*256
            cumulative_histogram = [0]*256
            total_pixels = self.img.width * self.img.height
            normalized_histogram = [0]*256
            normalized_cumulative_histogram = [0]*256

            # Calculate histogram
            for y in range(self.img.height):
                for x in range(self.img.width):
                    pixel_value = self.img.getpixel((x, y))
                    histogram[pixel_value] += 1

            # Calculate cumulative histogram 
            for i in range(256):
                cumulative_histogram[i] = sum(histogram[:i+1])

            # Calculate normalized histogram
            normalized_histogram = [count / total_pixels for count in histogram]

            # Calculate normalized cumulative histogram
            normalized_cumulative_histogram = [cumulative_histogram[i] / total_pixels for i in range(256)]

            # Plot histogram, cumulative histogram, normalized histogram, and normalized cumulative histogram side-by-side
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
            ax1.bar(range(256), histogram, color='black')
            ax1.set_title("Image Histogram")
            ax1.set_xlabel("Pixel Value")
            ax1.set_ylabel("Frequency")
            ax2.bar(range(256), cumulative_histogram, color='black')
            ax2.set_title("Cumulative Histogram")
            ax2.set_xlabel("Pixel Value")
            ax2.set_ylabel("Cumulative Frequency")
            ax3.bar(range(256), normalized_histogram, color='black')
            ax3.set_title("Normalized Histogram")
            ax3.set_xlabel("Pixel Value")
            ax3.set_ylabel("Normalized Frequency")
            ax4.bar(range(256), normalized_cumulative_histogram, color='black')
            ax4.set_title("Normalized Cumulative Histogram")
            ax4.set_xlabel("Pixel Value")
            ax4.set_ylabel("Normalized Cumulative Frequency")
            plt.show()
    
    def histogram_equalization(self):
        # Calculate histogram
        histogram = [0] * 256
        total_pixels = self.img.width * self.img.height
        
        for y in range(self.img.height):
            for x in range(self.img.width):
                pixel_value =self.img.getpixel((x, y))
                histogram[pixel_value] += 1

        # Calculate Normalized Cumulative histogram
        cdf = [0] * 256
        cumulative_sum = 0
        for i in range(256):
            cumulative_sum += histogram[i]
            cdf[i] = cumulative_sum / total_pixels

        # Calculate transformation function T(r)
        max = 256  # la valeur max
        transform = [round((max - 1) * cdf[i]) for i in range(256)]

        # Apply transformation to each pixel
        output_img = Image.new("L", self.img.size)
        for y in range(self.img.height):
            for x in range(self.img.width):
                pixel_value = self.img.getpixel((x, y))
                output_pixel = transform[pixel_value]
                output_img.putpixel((x, y), output_pixel)

        # Calculate and plot histogram of output image
        output_histogram = [0] * 256
        for y in range(output_img.height):
            for x in range(output_img.width):
                pixel_value = output_img.getpixel((x, y))
                output_histogram[pixel_value] += 1
        plt.bar(range(256), output_histogram, color='black')
        plt.title("Histogram of Output Image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

        # Display input and output images
        input_arr = np.array(self.img)
        output_arr = np.array(output_img)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(input_arr, cmap="gray")
        ax1.set_title("Input Image")
        ax2.imshow(output_arr, cmap="gray")
        ax2.set_title("Output Image")
        plt.show()

    def expansion_image(self):
        # Find the minimum and maximum pixel values of the input image
        min_value = np.min(self.img)
        max_value = np.max(self.img)
        min_value, max_value = self.img.getextrema()
        
        # Define the range of intensities to stretch the image to
        min = 0
        max = 255
        
        # Create output image with the same size and mode as input image
        output_img = Image.new(self.img.mode, self.img.size)
        
        # Apply the transformation function to each pixel of the input image
        for y in range(self.img.height):
            for x in range(self.img.width):
                pixel_value = self.img.getpixel((x,y))
                output_pixel = int((max / (max_value - min_value)) * (pixel_value - min_value))
                output_img.putpixel((x, y), output_pixel)
                
        # Display the input and output images using matplotlib
        plt.subplot(1,2,1)
        plt.imshow(self.img, cmap='gray')
        plt.title('Input Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1,2,2)
        plt.imshow(output_img, cmap='gray')
        plt.title('Output Image')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def translation_image(self, shift = 10):
        
        # Convert the image to grayscale
        gray_image = self.img.convert('L')
        
        # Get the pixel data as a list
        pixel_data = list(gray_image.getdata())
        
        # Apply the shift to the pixel values
        shifted_pixel_data = [(pixel + shift) % 256 for pixel in pixel_data]
        
        # Create a new image with the shifted pixel values
        shifted_image = Image.new('L', gray_image.size)
        shifted_image.putdata(shifted_pixel_data)
        
        # Create a subplot with two images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.img, cmap='gray')
        ax1.set_title('Original Image')
        ax2.imshow(shifted_image, cmap='gray')
        ax2.set_title('Translated Image') 
        plt.show()
    
    def histogram_inversion(self):
        # Convert the image to grayscale
        gray_image = self.img.convert('L')
        
        # Get the pixel data as a list
        pixel_data = list(gray_image.getdata())
        
        # Invert the pixel values
        inverted_pixel_data = [255 - pixel for pixel in pixel_data]
        
        # Create a new image with the inverted pixel values
        inverted_image = Image.new('L', gray_image.size)
        inverted_image.putdata(inverted_pixel_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.img, cmap='gray')
        ax1.set_title('Original Image')
        ax2.imshow(inverted_image, cmap='gray')
        ax2.set_title('Inverted Image')
        plt.show()
        
    def hitogram_colors(self):
        # Calculate histograms of the three color channels
        self.img =self.img.convert('RGB')
        b_hist = [0] * 256
        g_hist = [0] * 256
        r_hist = [0] * 256

        for y in range(self.img.height):
            for x in range(self.img.width):
                r, g, b = self.img.getpixel((x, y))
                b_hist[b] += 1
                g_hist[g] += 1
                r_hist[r] += 1

        # Plot the histograms using Matplotlib
        plt.figure()
        plt.plot(range(256), b_hist, color='blue', label='Blue')
        plt.plot(range(256), g_hist, color='green', label='Green')
        plt.plot(range(256), r_hist, color='red', label='Red')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def median_filter(self):

        # Convert the input image to grayscale
        img_gray = self.img.convert('L')

        kernel_size = 3 # Define the size of the kernel

        # Pad the input image with zeros to handle the edges
        padded_img = np.pad(np.array(img_gray), (kernel_size // 2, kernel_size // 2), 'constant')

        # Initialize the output image
        output_img = np.zeros_like(np.array(img_gray))

        # Iterate through each pixel in the input image
        for i in range(img_gray.size[0]):
            for j in range(img_gray.size[1]):
                # Create the sub-image (window)
                sub_img = padded_img[i:i+kernel_size, j:j+kernel_size]
                
                # Flatten the sub-image into a 1D array and sort it
                sorted_array = np.sort(sub_img.flatten())
                
                # Replace the pixel value with the median value of the sorted array
                output_img[i, j] = sorted_array[len(sorted_array) // 2]

        # Convert the output image to PIL format
        output_img_pil = Image.fromarray(output_img)

        # Display the input and output images using Matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_gray, cmap='gray')
        ax1.set_title('Input Image')
        ax2.imshow(output_img_pil, cmap='gray')
        ax2.set_title('Output Image')
        plt.show()

    def gaussian_filter(self):
        # Convert the input image to grayscale
        self.img = self.img.convert('L')

        # Define the size of the kernel
        kernel_size = 5

        # Define the standard deviation of the Gaussian kernel
        sigma = 1.0

        # Pad the input image with zeros to handle the edges
        pad_width = ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2))
        padded_img = np.pad(np.array(self.img), pad_width, 'constant')

        # Initialize the output image
        output_img = np.zeros_like(padded_img)

        # Define the Gaussian kernel
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)

        # Iterate through each pixel in the input image
        for i in range(kernel_size // 2, padded_img.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, padded_img.shape[1] - kernel_size // 2):
                # Create the sub-image (window)
                sub_img = padded_img[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1]

                # Convolve the sub-image with the Gaussian kernel
                output_img[i, j] = np.sum(sub_img * kernel)

        # Extract the central region of the filtered image (i.e., remove the zero-padding)
        filtered_img = output_img[kernel_size // 2:-kernel_size // 2, kernel_size // 2:-kernel_size // 2]

        # Display the original and filtered images
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(filtered_img, cmap='gray')
        ax[1].set_title('Filtered Image')
        ax[1].axis('off')
        plt.show()

    def sobel_filter(self):        
        
        # Convert the image to grayscale
        gray_img = self.img.convert('L')

        # Convert the grayscale image to a numpy array
        img_arr = np.array(gray_img)

        # Define the Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply the Sobel kernels using convolution
        grad_x = convolve2d(img_arr, sobel_x, mode='same')
        grad_y = convolve2d(img_arr, sobel_y, mode='same')

        # Compute the magnitude of the gradient
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize the gradient magnitude to the range [0, 255]
        grad_mag = (grad_mag / np.max(grad_mag)) * 255

        # Convert the gradient magnitude back to a PIL image
        grad_mag_img = Image.fromarray(grad_mag.astype(np.uint8))

        # Display the input and output images using Matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.img)
        ax1.set_title('Input Image')
        ax2.imshow(grad_mag_img)
        ax2.set_title('Sobel Filter Output')
        plt.show()

    def binarization(self):
        img = self.img.convert('L')
        h, w = img.size
        nb_pixels = h * w
        img2 = Image.new('L', img.size)

        #-----------binarisation--------------------#
        for i in range(h):
            for j in range(w):
                if img.getpixel((i, j)) > 170 and img.getpixel((i, j)) < 255 or img.getpixel((i, j)) > 20 and img.getpixel((i, j)) < 100:
                    img2.putpixel((i, j), 255)
                else:
                    img2.putpixel((i, j), 0)

        # Display images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image')
        ax2.imshow(img2, cmap='gray')
        ax2.set_title('Binarized Image')
        plt.show()
    
    def quantize_colors(self, n_colors=8):
        # Convert the image to an RGB numpy array
        self.img = self.img.convert('RGB')
        np_img = np.array(self.img)
        
        # Reshape the image array to a 2D array of RGB triplets
        pixel_values = np.reshape(np_img, (-1, 3))

        # Apply k-means clustering to quantize the colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixel_values)
        quantized_pixel_values = kmeans.cluster_centers_[kmeans.labels_]

        # Reshape the quantized pixels to the shape of the original image
        quantized_image = np.reshape(quantized_pixel_values, np_img.shape)

        # Scale the pixel values to the range [0, 255]
        quantized_image = (quantized_image * 255).astype(np.uint8)

        # Display the original image and the quantized image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(np_img)
        ax1.set_title('Original image')
        ax2.imshow(quantized_image)
        ax2.set_title('Quantized image with {} colors'.format(n_colors))
        plt.show()


root = tk.Tk()
my_gui = ImageHistogram(root)
root.mainloop()

