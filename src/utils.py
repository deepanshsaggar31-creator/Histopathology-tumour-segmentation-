import numpy as np

def normalize_stain_macenko(img, io=240, beta=0.15, alpha=1):
    """
    Stain normalization using Macenko method.
    """
    img = np.array(img, dtype=np.float32)
    img_clipped = np.clip(img, 1.0, 255.0)

    # 1. Convert RGB to Optical Density (OD)
    od = -np.log10(img_clipped / io)
    od_flat = od.reshape(-1, 3)

    # 2. Remove transparent background pixels
    mask = np.any(od_flat > beta, axis=1)
    od_hat = od_flat[mask]
    if len(od_hat) < 10:
        return img.astype(np.uint8)

    # 3. SVD decomposition to find color vectors
    cov = np.cov(od_hat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, [2, 1]]

    # 4. Project pixels and calculate angles
    proj = np.dot(od_hat, v)
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # 5. Extract extreme stain directions
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)
    v_h = np.dot(v, np.array([np.cos(min_angle), np.sin(min_angle)]))
    v_e = np.dot(v, np.array([np.cos(max_angle), np.sin(max_angle)]))

    stain_matrix = np.array([v_e, v_h]).T if v_h[0] < v_e[0] else np.array([v_h, v_e]).T

    # 6. Extract stain concentration maps
    stain_matrix_inv = np.linalg.pinv(stain_matrix)
    concentration = np.dot(stain_matrix_inv, od_flat.T)
    max_concentration = np.percentile(concentration, 99, axis=1, keepdims=True)

    # 7. Map to standard reference slides
    ref_stain_matrix = np.array([[0.5626, 0.2137], [0.7201, 0.8010], [0.4062, 0.5580]])
    ref_max_concentration = np.array([[1.9705], [1.0308]])

    normalized_concentration = concentration * (ref_max_concentration / (max_concentration + 1e-8))
    normalized_od = np.dot(ref_stain_matrix, normalized_concentration)

    # 8. Reconstruct image to RGB space
    normalized_img = io * np.power(10, -normalized_od)
    normalized_img = normalized_img.T.reshape(img.shape)

    return np.clip(normalized_img, 0.0, 255.0).astype(np.uint8)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plots train and val loss/accuracy history.
    """
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='orange')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='orange')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def plot_predictions(images, true_labels, pred_labels, title="Predictions Examples", save_path=None):
    """
    Helper to visualize batch predictions.
    """
    import matplotlib.pyplot as plt
    num_samples = len(images)
    if num_samples == 0:
        return
    fig, axes = plt.subplots(1, min(3, num_samples), figsize=(10, 3.5))
    if min(3, num_samples) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)
    for i in range(min(3, num_samples)):
        axes[i].imshow(images[i])
        axes[i].set_title(f"True: {true_labels[i]} | Pred: {pred_labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
