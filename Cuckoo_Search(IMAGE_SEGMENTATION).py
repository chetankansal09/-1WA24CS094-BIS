import subprocess
import sys

# --- AUTOMATIC LIBRARY INSTALLER ---
try:
    import numpy as np
    import cv2
except ModuleNotFoundError:
    print("Required libraries missing. Installing numpy and opencv-python automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "opencv-python"])
    import numpy as np
    import cv2
    print("Libraries installed successfully!\n")

import math

class CuckooSearchSegmentation:
    def __init__(self, n_nests=20, pa=0.25, max_iter=100, beta=1.5):
        self.n_nests = n_nests
        self.pa = pa
        self.max_iter = max_iter
        self.beta = beta
        
    def _compute_otsu_fitness(self, thresholds, histogram, total_pixels):
        thresholds = np.sort(np.clip(thresholds, 1, 254)).astype(int)
        thresholds = np.unique(thresholds)
        
        bins = np.concatenate(([0], thresholds, [255]))
        num_classes = len(bins) - 1
        
        weights = np.zeros(num_classes)
        means = np.zeros(num_classes)
        global_mean = 0
        
        for i in range(num_classes):
            low, high = bins[i], bins[i+1]
            weights[i] = np.sum(histogram[low:high]) / total_pixels
            if weights[i] > 0:
                intensities = np.arange(low, high)
                means[i] = np.sum(intensities * histogram[low:high]) / (weights[i] * total_pixels)
                global_mean += weights[i] * means[i]
        
        between_class_variance = np.sum(weights * (means - global_mean) ** 2)
        return between_class_variance

    def _levy_flight(self, size):
        sigma_num = math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        sigma_den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma_u = (sigma_num / sigma_den) ** (1 / self.beta)
        
        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)
        
        step = u / (np.abs(v) ** (1 / self.beta))
        return step

    def fit_segment(self, image_path, n_thresholds=3):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image from '{image_path}'. Place 'img.jpg' in the same folder.")
            
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        total_pixels = img.size
        
        nests = np.random.randint(1, 255, size=(self.n_nests, n_thresholds))
        fitness = np.array([self._compute_otsu_fitness(nest, hist, total_pixels) for nest in nests])
        
        best_idx = np.argmax(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for iteration in range(self.max_iter):
            steps = self._levy_flight(size=nests.shape)
            new_nests = nests + steps * (nests - best_nest)
            new_nests = np.clip(new_nests, 1, 254).astype(int)
            
            for i in range(self.n_nests):
                f_new = self._compute_otsu_fitness(new_nests[i], hist, total_pixels)
                if f_new > fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = f_new
                    
            discovered_mask = np.random.rand(self.n_nests) < self.pa
            if np.any(discovered_mask):
                perm_1 = np.random.permutation(self.n_nests)
                perm_2 = np.random.permutation(self.n_nests)
                
                step_size = np.random.rand(*nests.shape) * (nests[perm_1] - nests[perm_2])
                nests[discovered_mask] += step_size[discovered_mask].astype(int)
                nests = np.clip(nests, 1, 254).astype(int)
                
                fitness = np.array([self._compute_otsu_fitness(n, hist, total_pixels) for n in nests])
                
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_nest = nests[current_best_idx].copy()
                
        final_thresholds = np.sort(best_nest)
        segmented_img = np.zeros_like(img)
        
        vals = np.linspace(0, 255, n_thresholds + 1, dtype=np.uint8)
        for i in range(n_thresholds + 1):
            if i == 0:
                mask = img <= final_thresholds[i]
            elif i == n_thresholds:
                mask = img > final_thresholds[i-1]
            else:
                mask = (img > final_thresholds[i-1]) & (img <= final_thresholds[i])
            segmented_img[mask] = vals[i]
            
        return segmented_img, final_thresholds

# --- Execution ---
if __name__ == "__main__":
    csa_seg = CuckooSearchSegmentation(n_nests=25, pa=0.25, max_iter=50)
    image_name = "img.jpg"
    
    try:
        print(f"Processing '{image_name}' using Cuckoo Search Optimization...")
        segmented_output, thresholds = csa_seg.fit_segment(image_name, n_thresholds=3)
        
        print("\n--- Success! ---")
        print(f"Optimal Grayscale Thresholds Found: {thresholds}")
        
        # 1. Save standard grayscale segmentation map
        cv2.imwrite("segmented_grayscale.jpg", segmented_output)
        print("Saved: 'segmented_grayscale.jpg'")
        
        # 2. Save Vibrant False Color Map
        color_mapped = cv2.applyColorMap(segmented_output, cv2.COLORMAP_JET)
        cv2.imwrite("segmented_color.jpg", color_mapped)
        print("Saved: 'segmented_color.jpg' (Heatmap visualization)")
        
        # 3. Save Isolated Mask (Extracting the darkest segmented level)
        isolated_mask = np.where(segmented_output == 0, 255, 0).astype(np.uint8)
        cv2.imwrite("isolated_objects.jpg", isolated_mask)
        print("Saved: 'isolated_objects.jpg' (Binary mask isolating layer 0)")
        
        # 4. Extract Contours and Boundaries
        contours, _ = cv2.findContours(segmented_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Total distinct structures/edges identified: {len(contours)}")
        
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
