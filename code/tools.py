import numpy as np
from PIL import Image, ImageDraw, ImageFilter
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

def smooth_polyline_chaikin(points, iterations=6):
    """Lightly smooth a polyline using Chaikin's corner cutting while preserving endpoints."""
    if not points or len(points) < 3 or iterations <= 0:
        return points
    smoothed = points
    for _ in range(iterations):
        new_points = [smoothed[0]]  # preserve first endpoint
        for i in range(len(smoothed) - 1):
            x0, y0 = smoothed[i]
            x1, y1 = smoothed[i + 1]
            qx = 0.75 * x0 + 0.25 * x1
            qy = 0.75 * y0 + 0.25 * y1
            rx = 0.25 * x0 + 0.75 * x1
            ry = 0.25 * y0 + 0.75 * y1
            new_points.append((int(qx), int(qy)))
            new_points.append((int(rx), int(ry)))
        new_points.append(smoothed[-1])  # preserve last endpoint
        smoothed = new_points
    return smoothed

def generate_quadratic_bezier(p0, p1, p2, num_points):
    """Generate points along a quadratic Bezier curve defined by p0 (start), p1 (control), p2 (end)."""
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        one_minus = 1.0 - t
        x = one_minus * one_minus * p0[0] + 2 * one_minus * t * p1[0] + t * t * p2[0]
        y = one_minus * one_minus * p0[1] + 2 * one_minus * t * p1[1] + t * t * p2[1]
        points.append((int(x), int(y)))
    return points

def generate_cubic_bezier(p0, c1, c2, p3, num_points):
    """Generate points along a cubic Bezier curve defined by p0, c1, c2, p3."""
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = (u**3)*p0[0] + 3*(u**2)*t*c1[0] + 3*u*(t**2)*c2[0] + (t**3)*p3[0]
        y = (u**3)*p0[1] + 3*(u**2)*t*c1[1] + 3*u*(t**2)*c2[1] + (t**3)*p3[1]
        pts.append((int(x), int(y)))
    return pts

def _unit(vec):
    x, y = vec
    norm = (x*x + y*y) ** 0.5
    if norm == 0:
        return (0.0, 0.0)
    return (x / norm, y / norm)

def _top_point_and_tangent(line_points):
    """Return top-most point (min y) and a tangent vector at that point (toward interior)."""
    if not line_points:
        return None, (0.0, 1.0)
    # find index of min y
    idx = min(range(len(line_points)), key=lambda i: line_points[i][1])
    p_top = line_points[idx]
    # pick neighbor toward interior (larger y)
    if idx == 0:
        neighbor = line_points[1]
    elif idx == len(line_points) - 1:
        neighbor = line_points[-2]
    else:
        left = line_points[idx - 1]
        right = line_points[idx + 1]
        neighbor = left if left[1] > right[1] else right
    tangent = (neighbor[0] - p_top[0], neighbor[1] - p_top[1])
    return p_top, _unit(tangent)

def join_lines_at_top(lines, homography_matrix, original_shape, warped_shape, resize_value):
    """Join the palm lines at the top to create an M shape and smooth all segments."""
    if lines is None or len(lines) < 3:
        return []
    
    # Transform and smooth original lines
    transformed_lines = []
    for line in lines:
        if line is not None:
            transformed_line = transform_line_to_original_improved(line, homography_matrix, original_shape, warped_shape, resize_value)
            if transformed_line and len(transformed_line) > 1:
                smoothed_line = smooth_polyline_chaikin(transformed_line, iterations=6)
                transformed_lines.append(smoothed_line)
    
    if len(transformed_lines) < 3:
        return transformed_lines
    
    # Find top endpoints and their y-coordinates (closest to fingers = lowest y)
    top_info = []
    for line_pts in transformed_lines:
        p_top, tan = _top_point_and_tangent(line_pts)
        top_info.append((p_top, tan, p_top[1]))  # include y-coordinate for comparison
    
    # Find the line with the highest top endpoint (closest to fingers = lowest y value)
    # This is the line that extends furthest toward the fingers
    highest_line_idx = min(range(len(top_info)), key=lambda i: top_info[i][2])
    highest_line_top = top_info[highest_line_idx][0]
    highest_line_tan = top_info[highest_line_idx][1]
    
    # Connect other lines to the highest line's top endpoint
    connecting_lines = []
    for i in range(len(top_info)):
        if i != highest_line_idx:  # Skip the highest line itself
            p0, t0, y0 = top_info[i]
            
            # Connect this line to the highest line's top endpoint
            dx = highest_line_top[0] - p0[0]
            dy = highest_line_top[1] - p0[1]
            dist = max(1.0, (dx*dx + dy*dy) ** 0.5)
            
            # control handle length proportional to span
            handle = 0.3 * dist
            # lift upward a bit for nice arch
            lift = 0.2 * dist
            c1 = (int(p0[0] + t0[0]*handle), int(max(0, p0[1] + t0[1]*handle - lift)))
            c2 = (int(highest_line_top[0] - highest_line_tan[0]*handle), int(max(0, highest_line_top[1] - highest_line_tan[1]*handle - lift)))
            num_points = max(32, int(dist / 3))
            bezier = generate_cubic_bezier(p0, c1, c2, highest_line_top, num_points)
            # slight smoothing
            bezier = smooth_polyline_chaikin(bezier, iterations=2)
            connecting_lines.append(bezier)
    
    return transformed_lines + connecting_lines

def draw_polyline_rounded(draw, points, width, color):
    """Draw a thick polyline with rounded corners and endpoints (no overall blur)."""
    if not points or len(points) < 2:
        return
    # Draw thick line segments
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)
    # Draw circular caps at each vertex to round corners and endpoints
    r = max(1, width // 2)
    for (x, y) in points:
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=color)

def save_result_green_lines_on_original(original_image_path, warped_image_path, lines, resize_value, path_to_result):
    """Save result with green lines drawn on the original image"""
    if lines is None or len(lines) < 3:
        print_error()
        return
    
    original_img = cv2.imread(original_image_path)
    warped_img = cv2.imread(warped_image_path)
    if original_img is None or warped_img is None:
        print_error()
        return
    
    homography_matrix = get_homography_matrix(original_image_path, warped_image_path)
    if homography_matrix is None:
        print_error()
        return
    
    base = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    width = 20
    
    debug_warped_with_lines(warped_img, lines, resize_value, 'results/debug_warped_with_lines.jpg')
    all_lines = join_lines_at_top(lines, homography_matrix, original_img.shape, warped_img.shape, resize_value)
    
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    green = (0, 255, 0, 255)
    for line in all_lines:
        if line and len(line) > 1:
            draw_polyline_rounded(draw, line, width=width, color=green)
    
    # Remove global blur to avoid softening entire lines; corners are rounded by caps
    result = Image.alpha_composite(base, overlay)
    result.convert("RGB").save(path_to_result)

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