# Card Detector
## Card Detector: Implemented by image processing
* It can still be used even in low-light or highly reflective environments.
* Can be quickly implemented on other card types by taking a group of photos and marking the label.
<img src="https://user-images.githubusercontent.com/69178839/221424319-c2b368b5-02c3-44da-b625-753f69ca304f.jpg" width="900">
<img src="https://user-images.githubusercontent.com/69178839/221424320-2be7fa11-0b71-496e-83b4-0fe66a22fe99.jpg" width="900">
<img src="https://user-images.githubusercontent.com/69178839/221424322-8eb8f4e6-69e9-4350-b019-13b129cdc039.jpg" width="900">
<img src="https://user-images.githubusercontent.com/69178839/221424325-21032c28-c1c2-40ee-bdb3-755a61a2b2fc.jpg" width="900">

## Usage

* Take a photo of a group of cards and record the labels in the text file in sequence, from left to right in the photos.
* You can take more than one photo and more than one text file, just put them into `training_image_filename_list` and `training_labels_filename_list` in the file `card_detection.py` in order.
* The default is to take the corners of the picture to identify playing cards. If you want to identify other types of cards, you need to compare the entire card. It can be done by modifying the function `preprocess` in `card_detection.py`.
* Execute `card_detection.py`, the parameter is the photo to be identified. 
    ```bash
    python ./card_detection.py {photo path}
    ```

## Method
### Card Extraction
* Use the threshold value obtained by the Otsu algorithm to do binarization to avoid adverse light effects.
<img src="https://user-images.githubusercontent.com/69178839/221427277-e0fd53f0-d090-4c5d-a061-f81ae9968e24.jpg" width="900">
<img src="https://user-images.githubusercontent.com/69178839/221427787-6781ae2c-655d-4b69-a80d-02a7ccaca2d9.jpg" width="900">

### Correct the card
* Transform the slanted cards in the original image to rectangles by perspective transformation.  
    <img src="https://user-images.githubusercontent.com/69178839/221427849-95676ecd-e594-4eb2-b27d-dcebe120766a.png" width="25%"/>

### Take out the corner part and compare it with the labeled card
* Using a specially designed adaptive threshold, you can see that most of the information can be retained even if the reflection is severe.  
![Take out the corner part by specially designed adaptive threshold](https://user-images.githubusercontent.com/69178839/221427862-2b453131-35ae-4403-9471-94241afa3b08.png)
* Calculate the sum of the difference between the intensity value of each pixel of the labeled card and the query card. The label of the card with the smallest value difference is the recognition result.
