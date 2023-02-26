import sys
import numpy as np
import cv2
import os
from datetime import datetime

debug_mode = False

#####################################
# Reslut Image
#####################################
def draw_picture(ori_img, objects_dict, img_name = None, output_dir = "./result_image/"):
	output_img = ori_img.copy()
	scale = int(np.sqrt(objects_dict["area"]))

	for i in range(len(objects_dict["x"])):
		cX = int(objects_dict["x"][i])
		cY = int(objects_dict["y"][i])
		
		#draw 
		size = scale/150
		thickness = scale//80 + 1
		text = str(objects_dict["pattern"][i])
		text_size = cv2.getTextSize( text, cv2.FONT_HERSHEY_COMPLEX, size, thickness)
		text_width = text_size[0][0]
		text_height = text_size[0][1]
		output_img = cv2.putText(output_img, text, (cX - text_width//2, cY + text_height//2 ), cv2.FONT_HERSHEY_COMPLEX, size, (153, 255, 102), thickness, cv2.LINE_AA)
		   
	resize_img = cv2.resize(output_img,(1000, 1000*output_img.shape[0]//output_img.shape[1]))
	cv2.imshow(f'All detect', resize_img)
	cv2.waitKey(0)
	if output_dir is not None:
		os.makedirs( output_dir, exist_ok=True)			
		if img_name is None:
			currentTime = datetime.now()
			img_name = f"{currentTime.year}_{currentTime.month}_{currentTime.day}_{currentTime.hour}_{currentTime.minute}_{currentTime.second}"
		cv2.imwrite( output_dir + img_name + ".jpg", ori_img)
		cv2.imwrite( output_dir + img_name + "_detect.jpg", output_img)	
		
#####################################
# Matching labeled cards
#####################################
def preprocess(img):
	# Implement your own preprocessing
	thresh =  cornor_preprocess(img)
	#thresh =  whole_card_preprocess(img)
	return thresh

def whole_card_preprocess(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5, 5), 1)
	thresh = cv2.adaptiveThreshold(blur,255, 0, 0, 43, 3, cv2.THRESH_BINARY)
	return thresh

def cornor_preprocess(img, width_pattern_ratio = 0.2, height_rank_end_ratio = 0.18, height_suit_start_ratio = 0.14, height_suit_end_ratio = 0.28):
	(has_object1, top_left_rank) = get_corner(img, width_start = 0, width_end = width_pattern_ratio, height_start = 0, height_end = height_rank_end_ratio)
	(has_object2, top_left_suit) = get_corner(img, width_start = 0, width_end = width_pattern_ratio, height_start = height_suit_start_ratio, height_end = height_suit_end_ratio)
	(has_object3, button_right_rank) = get_corner(img, width_start = 1 - width_pattern_ratio, width_end = 1, height_start = 1 - height_rank_end_ratio, height_end = 1)
	(has_object4, button_right_suit) = get_corner(img, width_start = 1 - width_pattern_ratio, width_end = 1, height_start = 1 - height_suit_end_ratio, height_end = 1 - height_suit_start_ratio)

	output_img1 = np.concatenate((top_left_rank, top_left_suit))
	output_img2 = np.concatenate((button_right_rank, button_right_suit))
	output_img = np.concatenate((output_img1, output_img2), 1)

	return output_img

def get_corner(img1, width_start = 0, width_end = 0.19, height_start = 0.14, height_end = 0.26):
	has_object = True
	height = img1.shape[0]
	width = img1.shape[1]	

	crop_img_corner = img1[  int(height_start*height): int(height_end*height), int(width_start*width) : int(width_end*width) ]

	gray = cv2.cvtColor(crop_img_corner, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5, 5), 0 )
	thresh = cv2.adaptiveThreshold(blur, 255, 0, 1, 43, 3, cv2.THRESH_BINARY)

	query_suit_contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(query_suit_contours) > 0:
		query_suit_contours = sorted(query_suit_contours, key=cv2.contourArea,reverse=True)

		# Find bounding rectangle for largest contour, 
		# And use it to resize query suit image to match dimensions of the train suit image
		if len(query_suit_contours) != 0:
			x2,y2,w2,h2 = cv2.boundingRect(query_suit_contours[0])
			Qsuit_roi = thresh[y2:y2+h2, x2:x2+w2]
			Qsuit_sized = cv2.resize(Qsuit_roi, (100, 100), 0, 0)	
	else:
		has_object = False

	return has_object, 	Qsuit_sized	

def img_diff(img1, img2):
	blur1 = cv2.GaussianBlur(img1, (5,5), 5)
	blur2 = cv2.GaussianBlur(img2, (5,5), 5)		
	diff = cv2.absdiff(blur1, blur2)	
	diff = cv2.GaussianBlur(diff, (5,5), 5)
	
	flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
	diff_image = np.sum(diff)

	return diff_image 
	
def find_closest_card(training, img):
	features = preprocess(img.copy())
	if debug_mode: 
		cv2.imshow("test card: ", features)
		cv2.waitKey(0)
	closest = sorted(training.values(), key=lambda x:img_diff(x[1], features ))[0]
	
	if debug_mode: 
		cv2.imshow(f"closest card: {closest[0]}", closest[1])
		cv2.waitKey(0)
	
	return closest[0]
		 
#####################################
# Card Extraction
#####################################	
def getCards(im, max_cards = 100, mode = ""):
	thresholdArea = (im.shape[0]* im.shape[1]) // 100

	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#flag, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY) 
	 
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	object_num = 0
	object_contours = []
	for contour in contours:
		if cv2.contourArea(contour) > thresholdArea:
			object_num += 1
			object_contours.append(contour)

	contours = sorted(object_contours, key=lambda card:round(cv2.moments(card)["m10"] / cv2.moments(card)["m00"], 3))[:min(max_cards, len(contours))]
	img_contours = cv2.drawContours(im.copy(), contours, -1, (0,255,0), 5)
	
	if debug_mode: 
		if mode == "(Detect cards) ":
			cv2.imwrite( "Detect.jpg", img_contours)
		img_contours = cv2.resize(img_contours,(1000,600))
		cv2.imshow(f'{mode}There have {len(object_contours)} card',img_contours)
		cv2.waitKey(0)
	for card in contours:
		M = cv2.moments(card)
		cX = round(M["m10"] / M["m00"], 3)
		cY = round(M["m01"] / M["m00"], 3)

		peri = cv2.arcLength(card,True)
		
		'''cv2.imshow( "1", card)
		cv2.waitKey(0)'''
		rect = cv2.minAreaRect(card)
		approx = rectify(cv2.boxPoints(rect)) 		
		
		w = int(500)
		h = int(700)
		p = np.array([ [0,0],[w-1, 0],[w-1, h-1],[0, h-1] ],np.float32)
		#to vertical
		if distance(approx[0][0], approx[0][1], approx[1][0], approx[1][1]) >  distance(approx[1][0], approx[1][1], approx[2][0], approx[2][1]):
			p = np.array([ [w-1, 0],[w-1, h-1],[0, h-1], [0,0]],np.float32)
			#print("change to vertical")

		transform = cv2.getPerspectiveTransform(approx, p)
		warp = cv2.warpPerspective(im, transform, (w, h))
		area = cv2.contourArea(contour)
		yield warp, (cX, cY), area

def get_training(training_labels_filename_list, training_image_filename_list, avoid_cards=None):
	training = {}
	
	labels = {}
	label_count = 0
	for training_labels_filename in training_labels_filename_list:
		with open(training_labels_filename, encoding="utf8") as f:
			for line in f.readlines():
				num, suit = line.strip().split()
				labels[label_count] = (num,suit)
				label_count += 1

		
	print( "Training")
	training_image_list = []

	for training_image in training_image_filename_list:
		training_image_list.append(cv2.imread(training_image))
	#im = cv2.imread(training_image_filename_list)

	
	train_card_index = 0
	for training_image in training_image_list:
		for card, cXY, area in getCards( training_image, mode = "(Train cards) " ):
			if avoid_cards is None or (labels[train_card_index][0] not in avoid_cards[0] and labels[train_card_index][1] not in avoid_cards[1]):
				training[train_card_index] = (labels[train_card_index], preprocess(card))
				train_card_index += 1

	assert train_card_index == len(labels), "labels and cards do not match"

	print( "Done training")
	return training
	
def detect_cards(img, training):
	cards_pattern = []
	cards_cX = []
	cards_cY = []	
	area_list = []
	for i, (c, cXY, area) in enumerate(getCards(img, mode = "(Detect cards) ")):
		if debug_mode:
			cv2.imshow(f"detect card {i + 1}: ", c)
			cv2.waitKey(0)
		pattern = find_closest_card(training, c)
		cards_pattern.append(pattern)
		cards_cX.append(cXY[0])
		cards_cY.append(cXY[1])
		area_list.append(area)
		## mapping pattern to id
	
		print( f"card{i + 1}, Pattern: {cards_pattern[i]}, X: {cXY[0]}, Y: {cXY[1]}")

	mean_area = sum(area_list)/len(area_list) if len(area_list)>0 else 0
	return {"x": cards_cX, "y": cards_cY, "pattern": cards_pattern, "area": mean_area}

def rectify(h):
	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	max_d = 0
	index2 = -1
	for i in range(4):
		d = distance(h[np.argmin(add)][0], h[np.argmin(add)][1], h[i][0], h[i][1])
		if d > max_d:
			max_d = d
			index2 = i
	hnew[2] = h[index2]

	index1 = -1
	index3 = -1
	for i in range(4):
		if i != np.argmin(add) and i != index2 and index1 == -1:
			index1 = i
			continue
		if i != np.argmin(add) and i != index2:
			index3 = i
			break

	if h[index1][0] >= hnew[0][0] and h[index3][0] >= hnew[0][0]:
		if h[index1][1] < h[index3][1]:
			hnew[1] =  h[index1]
			hnew[3] =  h[index3]
		else:
			hnew[3] =  h[index1]
			hnew[1] =  h[index3]
	elif h[index1][0] >= hnew[0][0] and h[index3][0] <= hnew[0][0]:
		hnew[1] =  h[index1]
		hnew[3] =  h[index3]
	else:
		hnew[3] =  h[index1]
		hnew[1] =  h[index3]
		
	return hnew

def distance(x1, y1, x2, y2):
	return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )

if __name__ == '__main__':
	filename = sys.argv[1]
	training_image_filename_list = [ "card_data/train_S.jpg", "card_data/train_H.jpg", "card_data/train_D.jpg", "card_data/train_C.jpg", "card_data/train_1_7_another.jpg"] #add cards to training
	training_labels_filename_list = ["card_data/label.txt"]	
	
	training = get_training(training_labels_filename_list, training_image_filename_list)

	img = cv2.imread(filename)
	output_dict = detect_cards(img, training)

	draw_picture(img, output_dict )
