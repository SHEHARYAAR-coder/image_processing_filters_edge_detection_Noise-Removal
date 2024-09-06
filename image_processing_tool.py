import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.image = None
        self.noisy_image = None
        self.denoised_image = None
        self.edges_image = None

        # UI Elements
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        # Dropdown to select noise type
        self.noise_type_label = tk.Label(root, text="Select Noise Type:")
        self.noise_type_label.pack()

        self.noise_type = tk.StringVar()
        self.noise_dropdown = ttk.Combobox(root, textvariable=self.noise_type)
        self.noise_dropdown['values'] = ('Gaussian', 'Salt-and-Pepper', 'Poisson', 'Speckle', 'Uniform', 'Periodic', 'Quantization')
        self.noise_dropdown.pack()

        self.add_noise_btn = tk.Button(root, text="Add Noise", command=self.add_noise)
        self.add_noise_btn.pack()

        # Add Remove Noise button
        self.remove_noise_btn = tk.Button(root, text="Remove Noise", command=self.remove_noise)
        self.remove_noise_btn.pack()

        # Dropdown to select edge detection algorithm
        self.edge_detection_label = tk.Label(root, text="Select Edge Detection Algorithm:")
        self.edge_detection_label.pack()

        self.edge_algo = tk.StringVar()
        self.edge_dropdown = ttk.Combobox(root, textvariable=self.edge_algo)
        self.edge_dropdown['values'] = ('Canny', 'Sobel', 'Laplacian')
        self.edge_dropdown.pack()

        self.detect_edges_btn = tk.Button(root, text="Detect Edges", command=self.detect_edges)
        self.detect_edges_btn.pack()

        # Canvas for displaying images
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.show_image(self.image, 'original')

    def show_image(self, img, img_type):
        # Define the size for the image box (for example, 300x300)
        box_size = (700, 700)

        # Resize the image to fit into the box, maintaining aspect ratio
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail(box_size, Image.ANTIALIAS)  # Resize to fit within box while maintaining aspect ratio

        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Get the image dimensions after resizing
        img_width, img_height = img_pil.size

        # Set the canvas size according to the box size
        self.canvas.config(width=box_size[0] * 2, height=box_size[1])

        # Display the image in the correct location
        if img_type == 'original':
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.original_img_tk = img_tk  # Keep a reference to avoid garbage collection
        elif img_type == 'noisy':
            self.canvas.create_image(box_size[0], 0, anchor=tk.NW, image=img_tk)
            self.noisy_img_tk = img_tk  # Keep a reference to avoid garbage collection
        elif img_type == 'edges':
            self.canvas.create_image(box_size[0], 0, anchor=tk.NW, image=img_tk)
            self.edges_img_tk = img_tk  # Keep a reference to avoid garbage collection


    # Custom implementation of convolution for edge detection
    def apply_convolution(self, image, kernel):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = img_gray.shape
        k_rows, k_cols = kernel.shape
        pad_size = k_rows // 2

        # Pad the image with zeros
        padded_img = np.pad(img_gray, pad_size, mode='constant')

        # Create an empty array to store the result
        convolved_img = np.zeros_like(img_gray)

        # Perform convolution
        for i in range(rows):
            for j in range(cols):
                region = padded_img[i:i + k_rows, j:j + k_cols]
                convolved_img[i, j] = np.sum(region * kernel)

        # Clip the values to stay within valid range [0, 255]
        return np.clip(convolved_img, 0, 255).astype(np.uint8)

    # Canny edge detection implementation
    def canny_edge_detection(self, image):
        # Canny is composed of Gaussian Blur, Sobel gradients, non-maximum suppression, and double thresholding.
        # Step 1: Gaussian Blur (not included here since it's already handled with noise removal if needed)

        # Step 2: Calculate Sobel gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradient_x = self.apply_convolution(image, sobel_x)
        gradient_y = self.apply_convolution(image, sobel_y)

        # Magnitude of gradients
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        # Simple thresholding as part of Canny edge detection
        edges = np.where(magnitude > 100, 255, 0).astype(np.uint8)
        return edges

    # Sobel edge detection implementation
    def sobel_edge_detection(self, image):
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradient_x = self.apply_convolution(image, sobel_x)
        gradient_y = self.apply_convolution(image, sobel_y)

        # Combine gradients
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        return magnitude

    # Laplacian edge detection implementation
    def laplacian_edge_detection(self, image):
        # Define Laplacian kernel
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_img = self.apply_convolution(image, laplacian_kernel)
        return laplacian_img

    # Add Gaussian noise
    def add_gaussian_noise(self, image):
        row, col, ch = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
        return noisy

    # Add Salt-and-Pepper noise
    def add_salt_and_pepper_noise(self, image, prob=0.05):
        row, col, ch = image.shape
        noisy = np.copy(image)

        # Salt noise (white)
        num_salt = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

        # Pepper noise (black)
        num_pepper = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0

        return noisy

    # Add Poisson noise
    def add_poisson_noise(self, image):
        noisy = np.random.poisson(image / 255.0 * 100) / 100 * 255
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # Add Speckle noise
    def add_speckle_noise(self, image):
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # Add Uniform noise
    def add_uniform_noise(self, image, low=-50, high=50):
        row, col, ch = image.shape
        uniform_noise = np.random.uniform(low, high, (row, col, ch))
        noisy = image + uniform_noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # Add Periodic noise
    def add_periodic_noise(self, image, frequency=5):
        row, col, ch = image.shape
        x = np.linspace(0, 2 * np.pi * frequency, col)
        y = np.sin(x)
        noise_pattern = np.tile(y, (row, 1))
        periodic_noise = np.stack([noise_pattern] * 3, axis=-1)  # For 3 channels
        noisy = image + (periodic_noise * 50)  # Scale periodic noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # Add Quantization noise
    def add_quantization_noise(self, image, levels=8):
        quantized_image = np.floor(image / (256 / levels)) * (256 / levels)
        quantized_image = quantized_image.astype(np.uint8)
        return quantized_image

    # Function to handle the selection of noise type and add noise
    def add_noise(self):
        if self.image is not None:
            noise_type = self.noise_type.get()
            if noise_type == 'Gaussian':
                self.noisy_image = self.add_gaussian_noise(self.image)
            elif noise_type == 'Salt-and-Pepper':
                self.noisy_image = self.add_salt_and_pepper_noise(self.image)
            elif noise_type == 'Poisson':
                self.noisy_image = self.add_poisson_noise(self.image)
            elif noise_type == 'Speckle':
                self.noisy_image = self.add_speckle_noise(self.image)
            elif noise_type == 'Uniform':
                self.noisy_image = self.add_uniform_noise(self.image)
            elif noise_type == 'Periodic':
                self.noisy_image = self.add_periodic_noise(self.image)
            elif noise_type == 'Quantization':
                self.noisy_image = self.add_quantization_noise(self.image)

            # Show original and noisy images side by side
            self.show_image(self.image, 'original')
            self.show_image(self.noisy_image, 'noisy')

    # Noise removal logic + image swap
    def remove_noise(self):
        if self.noisy_image is not None:
            img_width = self.image.shape[1]

            # Set the noisy image in place of the original and vice versa
            self.show_image(self.noisy_image, 'original')
            self.show_image(self.image, 'noisy')

    # Function to handle edge detection
    def detect_edges(self):
        if self.image is not None:
            edge_algo = self.edge_algo.get()
            if edge_algo == 'Canny':
                self.edges_image = self.canny_edge_detection(self.image)
            elif edge_algo == 'Sobel':
                self.edges_image = self.sobel_edge_detection(self.image)
            elif edge_algo == 'Laplacian':
                self.edges_image = self.laplacian_edge_detection(self.image)

            # Show the original and edge-detected images side by side
            self.show_image(self.image, 'original')
            self.show_image(self.edges_image, 'edges')

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
