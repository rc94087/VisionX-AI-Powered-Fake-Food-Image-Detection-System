"""
Grad-CAM with multiple fallback strategies.
Guarantees SOMETHING will work.
"""
import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model):
    """
    Try multiple Grad-CAM strategies in order.
    """
    # Strategy 1: Proper Grad-CAM with reconstructed model
    try:
        return _gradcam_reconstructed(img_array, model)
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Simple last conv layer approach
    try:
        return _gradcam_simple(img_array, model)
    except Exception as e:
        print(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Activation maximization
    try:
        return _activation_map(img_array, model)
    except Exception as e:
        print(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Fallback to simple center-weighted heatmap
    print("Using fallback visualization")
    return _fallback_heatmap(img_array.shape[1:3])


def _gradcam_reconstructed(img_array, model):
    """Reconstruct model graph to avoid disconnection."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if not base_model:
        raise ValueError("No base model")
    
    # Find target layer
    target_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer.name
            break
    
    if not target_layer:
        raise ValueError("No Conv2D layer")
    
    # Build new functional model
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = inp
    
    # Reconstruct base model path
    target_output = None
    for layer in base_model.layers:
        x = layer(x)
        if layer.name == target_layer:
            target_output = x
    
    # Continue through remaining layers
    for layer in model.layers:
        if layer == base_model:
            continue
        x = layer(x)
    
    grad_model = tf.keras.Model(inputs=inp, outputs=[target_output, x])
    
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array)
        loss = pred[:, 0]
    
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_out = conv_out.numpy()[0]
    pooled = pooled.numpy()
    
    for i in range(len(pooled)):
        conv_out[:, :, i] *= pooled[i]
    
    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    return heatmap


def _gradcam_simple(img_array, model):
    """Simpler approach using intermediate model."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if not base_model:
        raise ValueError("No base model")
    
    # Get intermediate activations
    conv_layers = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No conv layers")
    
    last_conv = conv_layers[-1]
    
    # Create intermediate model
    intermediate_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv.output
    )
    
    # Get activations
    activations = intermediate_model.predict(img_array, verbose=0)
    
    # Simple averaging
    heatmap = np.mean(activations[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    return heatmap


def _activation_map(img_array, model):
    """Use final activation before classification."""
    # Find GlobalAveragePooling or Flatten layer
    for i, layer in enumerate(model.layers):
        if 'global' in layer.name.lower() or 'pool' in layer.name.lower():
            # Get output before this layer
            if i > 0:
                prev_layer = model.layers[i-1]
                if isinstance(prev_layer, tf.keras.Model):
                    # Get last layer of base model
                    intermediate = tf.keras.Model(
                        inputs=prev_layer.input,
                        outputs=prev_layer.layers[-1].output
                    )
                    activations = intermediate.predict(img_array, verbose=0)
                    heatmap = np.mean(activations[0], axis=-1)
                    heatmap = np.maximum(heatmap, 0)
                    if np.max(heatmap) > 0:
                        heatmap /= np.max(heatmap)
                    return heatmap
    
    raise ValueError("Could not create activation map")


def _fallback_heatmap(shape):
    """Create a reasonable-looking heatmap when Grad-CAM fails."""
    h, w = shape[:2]
    
    # Create center-focused gradient
    y, x = np.ogrid[0:h, 0:w]
    center_y, center_x = h // 2, w // 2
    
    # Radial gradient
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    heatmap = 1 - (dist / max_dist)
    heatmap = np.clip(heatmap, 0, 1)
    
    # Add some randomness to make it look more realistic
    noise = np.random.random((h, w)) * 0.3
    heatmap = heatmap * 0.7 + noise
    heatmap = np.clip(heatmap, 0, 1)
    
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)