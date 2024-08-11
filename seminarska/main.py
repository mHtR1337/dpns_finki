import cv2

video = cv2.VideoCapture('video.mp4')

background_subtractor = cv2.createBackgroundSubtractorMOG2()

vehicle_count = 0
line_position = 400

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = background_subtractor.apply(blurred)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if y > line_position - 5 and y < line_position + 5:
                vehicle_count += 1

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
