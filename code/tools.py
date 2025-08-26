import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener
import mediapipe as mp
import re

def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()  
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")

def remove_background(jpeg_dir, path_to_clean_image):
    if jpeg_dir[-4:] in ['heic', 'HEIC']:
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-4] + 'jpg')
        jpeg_dir = jpeg_dir[:-4] + 'jpg'
    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(result)  
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)
    img[mask == 255] = 255
    cv2.imwrite(path_to_clean_image, img)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
    else:
        heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2 = contents
        image_height, image_width = im.size
        fontsize = 12
        
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False
        )

        note_1 = '* Note: This program is just for fun! Please take the result with a light heart.'
        note_2 = '   If you want to check out more about palmistry, we recommend https://www.allure.com/story/palm-reading-guide-hand-lines'
        
        plt.title(' Check your palmistry result!', fontsize=14, y=1.01)

        plt.text(image_width + 15, 15, "<Heart line>", color='r', fontsize=fontsize)
        plt.text(image_width + 15, 35, heart_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 55, heart_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 80, "<Head line>", color='g', fontsize=fontsize)
        plt.text(image_width + 15, 100, head_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 120, head_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 145, "<Life line>", color='b', fontsize=fontsize)
        plt.text(image_width + 15, 165, life_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 185, life_content_2, fontsize=fontsize)

        plt.text(image_width + 15, 230, note_1, fontsize=fontsize-1, color='gray')
        plt.text(image_width + 15, 250, note_2, fontsize=fontsize-1, color='gray')

        plt.imshow(im)
        plt.savefig(path_to_result, bbox_inches = "tight")

def save_result_green_lines_only(im, lines, resize_value, path_to_result):
    """Save result with all lines in green color and no analysis text"""
    if im is None or lines is None or len(lines) < 3:
        print_error()
        return
    
    # Create a copy of the image to draw on
    im_copy = im.copy()
    draw = ImageDraw.Draw(im_copy)
    width = 3
    
    # Draw all lines in green
    for line in lines:
        if line is not None:
            line_points = [tuple(reversed(l[:2])) for l in line]
            draw.line(line_points, fill="green", width=width)
    
    # Save the image without any text analysis
    im_copy.save(path_to_result)

def join_lines_at_top(lines, homography_matrix, original_shape, warped_shape, resize_value):
    """Join the palm lines at the top to create an M shape"""
    if lines is None or len(lines) < 3:
        return []
    
    # Transform all lines to original coordinates
    transformed_lines = []
    for line in lines:
        if line is not None:
            transformed_line = transform_line_to_original_improved(line, homography_matrix, original_shape, warped_shape, resize_value)
            if transformed_line and len(transformed_line) > 1:
                transformed_lines.append(transformed_line)
    
    if len(transformed_lines) < 3:
        return transformed_lines
    
    # Find the top endpoints (lowest y coordinates - closest to fingers)
    top_endpoints = []
    for line in transformed_lines:
        # Find the point with minimum y coordinate (top of image)
        top_point = min(line, key=lambda p: p[1])
        top_endpoints.append(top_point)
    
    # Sort endpoints by x coordinate (left to right)
    top_endpoints.sort(key=lambda p: p[0])
    
    # Create connecting lines between adjacent endpoints
    connecting_lines = []
    for i in range(len(top_endpoints) - 1):
        start_point = top_endpoints[i]
        end_point = top_endpoints[i + 1]
        
        # Create a simple straight line between endpoints
        # Add some intermediate points for smoother connection
        num_points = max(5, int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) / 10))
        connecting_line = []
        
        for j in range(num_points + 1):
            t = j / num_points
            x = int(start_point[0] + t * (end_point[0] - start_point[0]))
            y = int(start_point[1] + t * (end_point[1] - start_point[1]))
            connecting_line.append((x, y))
        
        connecting_lines.append(connecting_line)
    
    # Return original lines plus connecting lines
    return transformed_lines + connecting_lines

def save_result_green_lines_on_original(original_image_path, warped_image_path, lines, resize_value, path_to_result):
    """Save result with green lines drawn on the original image"""
    if lines is None or len(lines) < 3:
        print_error()
        return
    
    # Load original and warped images
    original_img = cv2.imread(original_image_path)
    warped_img = cv2.imread(warped_image_path)
    
    if original_img is None or warped_img is None:
        print_error()
        return
    
    # Get homography matrix for inverse transformation
    homography_matrix = get_homography_matrix(original_image_path, warped_image_path)
    if homography_matrix is None:
        print_error()
        return
    
    # Create PIL image from original for drawing
    original_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(original_pil)
    width = 20  # Increased from 3 to 8 for thicker lines
    
    # Debug: Save warped image with lines for comparison
    debug_warped_with_lines(warped_img, lines, resize_value, 'results/debug_warped_with_lines.jpg')
    
    # Join lines at the top to create M shape
    all_lines = join_lines_at_top(lines, homography_matrix, original_img.shape, warped_img.shape, resize_value)
    
    # Draw all lines (original + connecting lines)
    for i, line in enumerate(all_lines):
        if line and len(line) > 1:
            draw.line(line, fill="green", width=width)
            # Debug: print some coordinate info
            if i < len(lines):  # Only print for original lines, not connecting lines
                print(f"Line {i}: {len(line)} points, first: {line[0]}, last: {line[-1]}")
    
    # Save the result
    original_pil.save(path_to_result)

def debug_warped_with_lines(warped_img, lines, resize_value, debug_path):
    """Debug function to save warped image with lines drawn on it"""
    import cv2
    debug_img = warped_img.copy()
    
    # Calculate scaling factors
    scale_x = warped_img.shape[1] / resize_value
    scale_y = warped_img.shape[0] / resize_value
    
    for line in lines:
        if line is not None:
            for y, x, _, _ in line:
                # Scale coordinates
                warped_x = int(x * scale_x)
                warped_y = int(y * scale_y)
                
                # Draw point on debug image
                if 0 <= warped_x < warped_img.shape[1] and 0 <= warped_y < warped_img.shape[0]:
                    cv2.circle(debug_img, (warped_x, warped_y), 2, (0, 255, 0), -1)
    
    cv2.imwrite(debug_path, debug_img)

def get_homography_matrix(original_image_path, warped_image_path):
    """Get the homography matrix used for warping"""
    # 7 landmark points (normalized) - same as in rectification.py
    pts_index = list(range(21))
    pts_target_normalized = np.float32([[1-0.48203104734420776, 0.9063420295715332],
                                        [1-0.6043621301651001, 0.8119394183158875],
                                        [1-0.6763232946395874, 0.6790258884429932],
                                        [1-0.7340714335441589, 0.5716733932495117],
                                        [1-0.7896472215652466, 0.5098430514335632],
                                        [1-0.5655680298805237, 0.5117031931877136],
                                        [1-0.5979393720626831, 0.36575648188591003],
                                        [1-0.6135331392288208, 0.2713503837585449],
                                        [1-0.6196483373641968, 0.19251111149787903],
                                        [1-0.4928809702396393, 0.4982593059539795],
                                        [1-0.4899863600730896, 0.3213786780834198],
                                        [1-0.4894656836986542, 0.21283167600631714],
                                        [1-0.48334982991218567, 0.12900274991989136],
                                        [1-0.4258815348148346, 0.5180916786193848],
                                        [1-0.4033462107181549, 0.3581996262073517],
                                        [1-0.3938145041465759, 0.2616880536079407],
                                        [1-0.38608720898628235, 0.1775170862674713],
                                        [1-0.36368662118911743, 0.5642163157463074],
                                        [1-0.33553171157836914, 0.44737303256988525],
                                        [1-0.3209102153778076, 0.3749568462371826],
                                        [1-0.31213682889938354, 0.3026996850967407]])
    
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Extract 21 landmark points from original image
        image = cv2.flip(cv2.imread(original_image_path), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = image.shape
        
        if results.multi_hand_landmarks is None:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                        hand_landmarks.landmark[i].y*image_height] for i in pts_index])
        pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
        
        M, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC, 5.0)
        return M

def transform_line_to_original_improved(line, homography_matrix, original_shape, warped_shape, resize_value):
    """Improved transformation with better coordinate handling"""
    if homography_matrix is None:
        return None
    
    # Get inverse homography matrix
    inv_matrix = np.linalg.inv(homography_matrix)
    
    original_points = []
    
    # Calculate scaling factors more precisely
    scale_x = warped_shape[1] / resize_value
    scale_y = warped_shape[0] / resize_value
    
    for y, x, _, _ in line:
        # Scale from detection space (256x256) to warped image space
        warped_x = x * scale_x
        warped_y = y * scale_y
        
        # Ensure coordinates are within warped image bounds
        warped_x = max(0, min(warped_x, warped_shape[1] - 1))
        warped_y = max(0, min(warped_y, warped_shape[0] - 1))
        
        # Transform point from warped to flipped original coordinates
        point = np.array([[warped_x, warped_y, 1]], dtype=np.float32)
        transformed = inv_matrix.dot(point.T)
        transformed = transformed / transformed[2]  # Normalize homogeneous coordinates
        
        # Convert to integer coordinates (these are in flipped image space)
        flipped_x = int(round(float(transformed[0])))
        flipped_y = int(round(float(transformed[1])))
        
        # Apply inverse horizontal flip to get original image coordinates
        orig_x = original_shape[1] - 1 - flipped_x  # Horizontal flip
        orig_y = flipped_y  # Y coordinate stays the same
        
        # Check if point is within original image bounds
        if 0 <= orig_x < original_shape[1] and 0 <= orig_y < original_shape[0]:
            original_points.append((orig_x, orig_y))
    
    # Filter out duplicate consecutive points and ensure minimum line length
    filtered_points = []
    for i, point in enumerate(original_points):
        if i == 0 or point != original_points[i-1]:
            filtered_points.append(point)
    
    return filtered_points if len(filtered_points) > 1 else None

def print_error():
    print('Palm lines not properly detected! Please use another palm image.')