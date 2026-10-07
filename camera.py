import cv2
import mediapipe as mp
import time
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# SELFIE SEGMENTATION
# ============================================================

segmenter = None

try:
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(
            model_asset_path='selfie_segmenter.tflite'
        ),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True
    )

    segmenter = ImageSegmenter.create_from_options(
        seg_options
    )

except Exception as e:

    print(
        "Segmenter is not available "
        "(make sure selfie_segmenter.tflite exists):",
        e
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


# ============================================================
# GENERATE GALAXY BACKGROUND
# ============================================================

galaxy_bg = np.zeros(
    (1080, 1920, 3),
    dtype=np.uint8
)

galaxy_bg[:] = (
    30,
    10,
    40
)

# Small stars
for _ in range(800):

    sx = np.random.randint(
        0,
        1920
    )

    sy = np.random.randint(
        0,
        1080
    )

    galaxy_bg[
        sy,
        sx
    ] = (
        255,
        255,
        255
    )


# Larger colorful stars
for _ in range(100):

    sx = np.random.randint(
        0,
        1920
    )

    sy = np.random.randint(
        0,
        1080
    )

    cv2.circle(
        galaxy_bg,
        (sx, sy),
        np.random.randint(2, 6),
        (
            np.random.randint(150, 255),
            np.random.randint(100, 255),
            255
        ),
        -1
    )


print(
    "Opening camera... "
    "Press 'q' on the keyboard to exit."
)


# ============================================================
# FILTERS
# ============================================================

filters = [
    "MONO",
    "DUAL-TONE",
    "PIXELATE",
    "INVERT",
    "SEPIA",
    "BLUR",
    "THERMAL",
    "SKETCH",
    "GLITCH",
    "NEON",
    "GALAXY"
]

current_filter = 0
gesture_triggered = False


# ============================================================
# APPLY FILTER
# ============================================================

def apply_filter(
    roi,
    filter_name,
    x=0,
    y=0,
    mask_person=None,
    frame_galaxy=None
):

    # --------------------------------------------------------
    # MONO
    # --------------------------------------------------------

    if filter_name == "MONO":

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )


    # --------------------------------------------------------
    # INVERT
    # --------------------------------------------------------

    elif filter_name == "INVERT":

        return cv2.bitwise_not(
            roi
        )


    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    elif filter_name == "BLUR":

        return cv2.GaussianBlur(
            roi,
            (25, 25),
            0
        )


    # --------------------------------------------------------
    # SEPIA
    # --------------------------------------------------------

    elif filter_name == "SEPIA":

        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])

        filtered = cv2.transform(
            roi,
            kernel
        )

        return np.clip(
            filtered,
            0,
            255
        ).astype(np.uint8)


    # --------------------------------------------------------
    # DUAL-TONE
    # --------------------------------------------------------

    elif filter_name == "DUAL-TONE":

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        _, mask_c = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        filtered = np.zeros_like(
            roi
        )

        # Orange
        filtered[
            mask_c == 255
        ] = [
            0,
            165,
            255
        ]

        # Pink
        filtered[
            mask_c == 0
        ] = [
            147,
            20,
            255
        ]

        return filtered


    # --------------------------------------------------------
    # PIXELATE
    # --------------------------------------------------------

    elif filter_name == "PIXELATE":

        h_r, w_r = roi.shape[:2]

        if h_r > 10 and w_r > 10:

            small = cv2.resize(
                roi,
                (
                    w_r // 10,
                    h_r // 10
                ),
                interpolation=cv2.INTER_LINEAR
            )

            return cv2.resize(
                small,
                (w_r, h_r),
                interpolation=cv2.INTER_NEAREST
            )


    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    elif filter_name == "THERMAL":

        return cv2.applyColorMap(
            roi,
            cv2.COLORMAP_JET
        )


    # --------------------------------------------------------
    # SKETCH
    # --------------------------------------------------------

    elif filter_name == "SKETCH":

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        inv = cv2.bitwise_not(
            gray
        )

        blur = cv2.GaussianBlur(
            inv,
            (21, 21),
            0
        )

        sketch = cv2.divide(
            gray,
            255 - blur,
            scale=256
        )

        return cv2.cvtColor(
            sketch,
            cv2.COLOR_GRAY2BGR
        )


    # --------------------------------------------------------
    # GLITCH
    # --------------------------------------------------------

    elif filter_name == "GLITCH":

        h_r, w_r = roi.shape[:2]

        shift = max(
            5,
            w_r // 20
        )

        glitch_roi = roi.copy()

        if w_r > shift:

            glitch_roi[
                :,
                :-shift,
                2
            ] = roi[
                :,
                shift:,
                2
            ]

            glitch_roi[
                :,
                shift:,
                0
            ] = roi[
                :,
                :-shift,
                0
            ]

        return glitch_roi


    # --------------------------------------------------------
    # NEON
    # --------------------------------------------------------

    elif filter_name == "NEON":

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        edges_bgr = cv2.cvtColor(
            edges,
            cv2.COLOR_GRAY2BGR
        )

        edges_bgr[
            np.where(
                (
                    edges_bgr ==
                    [255, 255, 255]
                ).all(axis=2)
            )
        ] = [
            255,
            255,
            0
        ]

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        return cv2.dilate(
            edges_bgr,
            kernel,
            iterations=1
        )


    # --------------------------------------------------------
    # GALAXY
    # --------------------------------------------------------

    elif (
        filter_name == "GALAXY"
        and mask_person is not None
        and frame_galaxy is not None
    ):

        bh, bw = roi.shape[:2]

        roi_mask = mask_person[
            y:y + bh,
            x:x + bw
        ]

        roi_galaxy = frame_galaxy[
            y:y + bh,
            x:x + bw
        ]

        bg_condition = (
            roi_mask == 0
        )

        filtered = roi.copy()

        filtered[
            bg_condition
        ] = roi_galaxy[
            bg_condition
        ]

        return filtered


    return roi


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    success, img = cap.read()

    if not success:
        break

    # Mirror the camera
    img = cv2.flip(
        img,
        1
    )

    h, w, c = img.shape


    # ========================================================
    # RESIZE GALAXY BACKGROUND
    # ========================================================

    frame_galaxy = galaxy_bg[
        :h,
        :w
    ]


    # ========================================================
    # CONVERT TO MEDIAPIPE IMAGE
    # ========================================================

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    timestamp_ms = (
        time.time_ns() // 1_000_000
    )


    # ========================================================
    # DETECT HANDS
    # ========================================================

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    filter_name = filters[
        current_filter
    ]


    # ========================================================
    # PERSON SEGMENTATION
    # ========================================================

    mask_person = None

    if (
        filter_name == "GALAXY"
        and segmenter is not None
    ):

        seg_result = (
            segmenter.segment_for_video(
                mp_image,
                timestamp_ms
            )
        )

        if seg_result.category_mask is not None:

            mask_person = (
                seg_result.category_mask.numpy_view()
            )

            if mask_person.shape != (
                h,
                w
            ):

                mask_person = cv2.resize(
                    mask_person,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )


    # ========================================================
    # HAND PORTAL
    # ========================================================

    pts_portal = []

    change_filter = False


    if results.hand_landmarks:

        # ====================================================
        # CHECK GESTURE TO CHANGE FILTER
        # ====================================================

        if len(
            results.hand_landmarks
        ) >= 2:

            idx0 = (
                results.hand_landmarks[0][8]
            )

            idx1 = (
                results.hand_landmarks[1][8]
            )

            pt0 = (
                int(idx0.x * w),
                int(idx0.y * h)
            )

            pt1 = (
                int(idx1.x * w),
                int(idx1.y * h)
            )

            # Index fingers close together
            if math.hypot(
                pt0[0] - pt1[0],
                pt0[1] - pt1[1]
            ) < 40:

                change_filter = True


        # ====================================================
        # BACKUP FILTER GESTURE
        # THUMB + PINKY
        # ====================================================

        for hand_lms in results.hand_landmarks:

            thumb = hand_lms[4]

            pinky = hand_lms[20]

            tx = int(
                thumb.x * w
            )

            ty = int(
                thumb.y * h
            )

            px = int(
                pinky.x * w
            )

            py = int(
                pinky.y * h
            )

            if math.hypot(
                tx - px,
                ty - py
            ) < 40:

                change_filter = True


        # ====================================================
        # CHANGE FILTER
        # ====================================================

        if change_filter:

            if not gesture_triggered:

                current_filter = (
                    current_filter + 1
                ) % len(filters)

                gesture_triggered = True

        else:

            gesture_triggered = False


        # ====================================================
        # COLLECT PORTAL POINTS
        # ====================================================

        for hand_lms in results.hand_landmarks:

            for id, lm in enumerate(
                hand_lms
            ):

                cx = int(
                    lm.x * w
                )

                cy = int(
                    lm.y * h
                )

                # Thumb + index fingertips
                if id in [4, 8]:

                    pts_portal.append([
                        cx,
                        cy
                    ])

                    cv2.circle(
                        img,
                        (cx, cy),
                        8,
                        (255, 255, 0),
                        cv2.FILLED
                    )


        # ====================================================
        # CREATE 4-POINT PORTAL
        # ====================================================

        if len(
            pts_portal
        ) == 4:

            # Sort points by Y
            pts_portal.sort(
                key=lambda p: p[1]
            )

            top_pts = pts_portal[:2]

            bottom_pts = pts_portal[2:]


            # Sort top points by X
            top_pts.sort(
                key=lambda p: p[0]
            )

            # Sort bottom points by X
            bottom_pts.sort(
                key=lambda p: p[0]
            )


            # Create polygon
            poly_pts = np.array(
                [
                    top_pts[0],
                    top_pts[1],
                    bottom_pts[1],
                    bottom_pts[0]
                ],
                dtype=np.int32
            )


            # =================================================
            # PORTAL BOUNDING BOX
            # =================================================

            x, y, bw, bh = (
                cv2.boundingRect(
                    poly_pts
                )
            )

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            bw = min(
                w - x,
                bw
            )

            bh = min(
                h - y,
                bh
            )


            if bw > 0 and bh > 0:

                # Get camera region
                roi = img[
                    y:y + bh,
                    x:x + bw
                ].copy()


                # Apply selected filter
                filtered_roi = apply_filter(
                    roi,
                    filter_name,
                    x,
                    y,
                    mask_person,
                    frame_galaxy
                )


                # =================================================
                # CREATE PORTAL MASK
                # =================================================

                mask = np.zeros(
                    (bh, bw),
                    dtype=np.uint8
                )

                poly_roi = (
                    poly_pts -
                    [x, y]
                )

                cv2.fillPoly(
                    mask,
                    [poly_roi],
                    255
                )


                mask_3ch = cv2.cvtColor(
                    mask,
                    cv2.COLOR_GRAY2BGR
                )


                # Apply filter only inside portal
                img[
                    y:y + bh,
                    x:x + bw
                ] = np.where(
                    mask_3ch == 255,
                    filtered_roi,
                    roi
                )


                # =================================================
                # PORTAL BORDER
                # =================================================

                cv2.polylines(
                    img,
                    [poly_pts],
                    True,
                    (255, 255, 255),
                    2
                )


                # =================================================
                # GLOW PARTICLES
                # =================================================

                for i in range(4):

                    pt1 = poly_pts[i]

                    pt2 = poly_pts[
                        (i + 1) % 4
                    ]

                    for _ in range(5):

                        alpha = np.random.random()

                        particle_x = int(
                            pt1[0] * alpha
                            +
                            pt2[0] * (1 - alpha)
                        ) + np.random.randint(
                            -15,
                            15
                        )

                        particle_y = int(
                            pt1[1] * alpha
                            +
                            pt2[1] * (1 - alpha)
                        ) + np.random.randint(
                            -15,
                            15
                        )

                        cv2.circle(
                            img,
                            (
                                particle_x,
                                particle_y
                            ),
                            np.random.randint(
                                1,
                                4
                            ),
                            (0, 255, 255),
                            -1
                        )


                # =================================================
                # PORTAL LABEL
                # =================================================

                cv2.putText(
                    img,
                    f"PORTAL: {filter_name}",
                    (
                        top_pts[0][0],
                        top_pts[0][1] - 10
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 255, 255),
                    2
                )


    # ========================================================
    # SCREEN TEXT
    # ========================================================

    cv2.putText(
        img,
        "Change Filter: Thumb & Pinky",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"Active Filter: {filters[current_filter]}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "RETROLENS - Python",
        img
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()